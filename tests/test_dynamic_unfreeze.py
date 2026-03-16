"""
Unit tests for the Resolution Probe and Dynamic Unfreezing system.

Covers:
    - get_resolution_probe()      : forward-hook stage detection
    - get_unfreeze_units()        : CNN path + isotropic ViT path
    - thaw_units()                : requires_grad toggling
    - DynamicThawController       : plateau detection, LR decay, edge cases
    - create_optimizer() dynamic mode : classifier-only startup
"""

import sys
import os
import unittest

# Make sure imports resolve from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torchvision import models

from utils.model_utils import (
    get_resolution_probe,
    get_unfreeze_units,
    thaw_units,
    UnfreezeUnit,
    freeze_backbone,
    load_model,
    _get_classifier_attr,
)
from utils.trainer import DynamicThawController, create_optimizer


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


class TinyCNN(nn.Module):
    """
    Minimal CNN with predictable resolution stages.

    Input 16x16 → Stage 0 (16x16) → Stage 1 (8x8) → Stage 2 (4x4) → head.

    Leaf module outputs at each resolution:
        16x16: conv_a, conv_b
         8x8 : pool_a, conv_c
         4x4 : pool_b, conv_d  (also avgpool at 1x1 in final flush)
         1x1 : avgpool
    """

    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(3, 4, 3, padding=1)  # 16x16
        self.conv_b = nn.Conv2d(4, 4, 3, padding=1)  # 16x16
        self.pool_a = nn.MaxPool2d(2)  # 8x8
        self.conv_c = nn.Conv2d(4, 8, 3, padding=1)  # 8x8
        self.pool_b = nn.MaxPool2d(2)  # 4x4
        self.conv_d = nn.Conv2d(8, 8, 1)  # 4x4
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 1x1
        self.fc = nn.Linear(8, 4)  # head (2D, not probed)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.pool_a(x)
        x = self.conv_c(x)
        x = self.pool_b(x)
        x = self.conv_d(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.fc(x)


def _make_fake_units(n):
    """Return n synthetic UnfreezeUnit objects."""
    return [
        UnfreezeUnit(
            stage_id=i, module_names=[f"layer{i}"], parameter_count=100 * (i + 1)
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Resolution Probe
# ---------------------------------------------------------------------------


class TestResolutionProbe(unittest.TestCase):

    def setUp(self):
        self.model = TinyCNN()
        self.input_shape = (3, 16, 16)

    def test_returns_list_of_dicts(self):
        result = get_resolution_probe(self.model, self.input_shape)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        for stage in result:
            self.assertIn("stage_id", stage)
            self.assertIn("resolution", stage)
            self.assertIn("module_names", stage)
            self.assertIn("parameter_count", stage)

    def test_stage_ids_are_sequential(self):
        result = get_resolution_probe(self.model, self.input_shape)
        ids = [s["stage_id"] for s in result]
        self.assertEqual(ids, list(range(len(ids))))

    def test_resolutions_are_tuples(self):
        result = get_resolution_probe(self.model, self.input_shape)
        for stage in result:
            res = stage["resolution"]
            self.assertIsInstance(res, tuple)
            self.assertEqual(len(res), 2)

    def test_resolution_changes_between_stages(self):
        result = get_resolution_probe(self.model, self.input_shape)
        resolutions = [s["resolution"] for s in result]
        # All resolutions should be distinct (each stage has a different H×W)
        self.assertEqual(len(resolutions), len(set(resolutions)))

    def test_detects_expected_stages_for_tiny_cnn(self):
        """TinyCNN has resolution steps 16→8→4→1, so 4 stages."""
        result = get_resolution_probe(self.model, self.input_shape)
        resolutions = {s["resolution"] for s in result}
        self.assertIn((16, 16), resolutions)
        self.assertIn((8, 8), resolutions)
        self.assertIn((4, 4), resolutions)
        self.assertIn((1, 1), resolutions)

    def test_module_names_are_non_empty_strings(self):
        result = get_resolution_probe(self.model, self.input_shape)
        for stage in result:
            self.assertTrue(len(stage["module_names"]) > 0)
            for name in stage["module_names"]:
                self.assertIsInstance(name, str)
                self.assertTrue(len(name) > 0)

    def test_parameter_count_is_positive_or_zero(self):
        result = get_resolution_probe(self.model, self.input_shape)
        for stage in result:
            self.assertGreaterEqual(stage["parameter_count"], 0)

    def test_conv_a_and_conv_b_in_same_stage(self):
        """conv_a and conv_b both output 16x16, so they share a stage."""
        result = get_resolution_probe(self.model, self.input_shape)
        stage_16 = next(s for s in result if s["resolution"] == (16, 16))
        names = stage_16["module_names"]
        self.assertIn("conv_a", names)
        self.assertIn("conv_b", names)

    def test_hooks_are_removed_after_probe(self):
        """After the probe, no forward hooks should remain on the model."""
        get_resolution_probe(self.model, self.input_shape)
        hook_count = sum(len(m._forward_hooks) for m in self.model.modules())
        self.assertEqual(hook_count, 0)

    def test_model_remains_in_eval_mode(self):
        self.model.train()
        get_resolution_probe(self.model, self.input_shape)
        self.assertFalse(self.model.training)

    def test_resnet50_has_multiple_stages(self):
        """ResNet50 (no weights) should probe to ≥4 resolution stages."""
        resnet = models.resnet50(weights=None)
        result = get_resolution_probe(resnet, (3, 224, 224))
        self.assertGreaterEqual(len(result), 4)

    def test_resnet50_stage_7x7_present(self):
        resnet = models.resnet50(weights=None)
        result = get_resolution_probe(resnet, (3, 224, 224))
        resolutions = {s["resolution"] for s in result}
        self.assertIn((7, 7), resolutions)


# ---------------------------------------------------------------------------
# 2. get_unfreeze_units
# ---------------------------------------------------------------------------


class TestGetUnfreezeUnits(unittest.TestCase):

    def _make_resnet50(self):
        """ResNet50 with custom 4-class head, backbone frozen."""
        options = {
            "model": {
                "backbone": "resnet50",
                "pretrained": False,
                "freeze_backbone": True,
                "classifier_hidden": [],
                "dropout": 0.0,
            }
        }
        return load_model(options, num_classes=4)

    def test_returns_list_of_unfreeze_units(self):
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))
        self.assertIsInstance(units, list)
        for u in units:
            self.assertIsInstance(u, UnfreezeUnit)

    def test_units_not_empty(self):
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))
        self.assertGreater(len(units), 0)

    def test_classifier_excluded_from_units(self):
        """No unit should contain the 'fc' classifier module."""
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))
        all_names = [name for u in units for name in u.module_names]
        self.assertFalse(any(name == "fc" for name in all_names))

    def test_ordered_output_to_input(self):
        """For ResNet50, the first unit (output→input) should contain layer4
        modules — layer4 is the last backbone stage before avgpool/fc."""
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        # All module names across all units
        all_names = [name for u in units for name in u.module_names]

        # layer4 modules should appear somewhere in the units list
        self.assertTrue(
            any("layer4" in name for name in all_names),
            "Expected layer4 modules in unfreeze units",
        )

        # The first unit (closest to head) should contain layer4 modules
        # because layer4 is the last (output-most) backbone stage in ResNet
        first_unit_names = units[0].module_names
        self.assertTrue(
            any("layer4" in name for name in first_unit_names),
            f"Expected units[0] to contain layer4 modules, got: {first_unit_names[:3]}",
        )

    def test_parameter_counts_positive(self):
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))
        for u in units:
            self.assertGreater(u.parameter_count, 0)

    def test_vit_isotropic_path(self):
        """ViT-B/16 should use the isotropic registry and return 12 units."""
        vit = models.vit_b_16(weights=None)
        # Attach a trivial head so _get_classifier_attr works
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16", (3, 224, 224))
        self.assertEqual(len(units), 12)

    def test_vit_units_are_transformer_blocks(self):
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16", (3, 224, 224))
        # All module_names should be of the form "encoder.layers.N"
        for u in units:
            for name in u.module_names:
                self.assertTrue(
                    name.startswith("encoder.layers."),
                    f"Expected encoder.layers.N, got {name!r}",
                )

    def test_vit_ordered_output_to_input(self):
        """First unit should be block 11 (last block = closest to head)."""
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16", (3, 224, 224))
        self.assertIn("encoder.layers.11", units[0].module_names)
        self.assertIn("encoder.layers.0", units[-1].module_names)


# ---------------------------------------------------------------------------
# 3. thaw_units
# ---------------------------------------------------------------------------


class TestThawUnits(unittest.TestCase):

    def _frozen_resnet50(self):
        options = {
            "model": {
                "backbone": "resnet50",
                "pretrained": False,
                "freeze_backbone": True,
                "classifier_hidden": [],
                "dropout": 0.0,
            }
        }
        return load_model(options, num_classes=4)

    def test_thaw_increases_trainable_params(self):
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        thaw_units(model, [units[0]])
        after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(after, before)

    def test_thaw_returns_correct_count(self):
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        n = thaw_units(model, [units[0]])
        self.assertGreater(n, 0)
        # Count should equal new trainable params from that unit
        unit_params = sum(
            p.numel()
            for name in units[0].module_names
            for p in model.get_submodule(name).parameters()
        )
        self.assertEqual(n, unit_params)

    def test_thaw_sets_requires_grad_true(self):
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        unit = units[0]
        thaw_units(model, [unit])
        for name in unit.module_names:
            for param in model.get_submodule(name).parameters():
                self.assertTrue(
                    param.requires_grad,
                    f"Param in {name!r} still frozen after thaw_units()",
                )

    def test_thaw_is_idempotent(self):
        """Calling thaw_units twice on the same unit should not double-count."""
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        thaw_units(model, [units[0]])
        trainable_after_first = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Second call returns 0 since all are already unfrozen
        returned = thaw_units(model, [units[0]])
        trainable_after_second = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        self.assertEqual(returned, 0)
        self.assertEqual(trainable_after_first, trainable_after_second)

    def test_other_units_remain_frozen(self):
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        # Thaw only the first unit
        thaw_units(model, [units[0]])

        # Collect all module names thawed
        thawed_names = set(units[0].module_names)

        # Modules from other units should still be frozen
        for other_unit in units[1:]:
            for name in other_unit.module_names:
                for param in model.get_submodule(name).parameters():
                    self.assertFalse(
                        param.requires_grad,
                        f"Param in {name!r} was unexpectedly unfrozen",
                    )

    def test_full_gradual_unfreeze_simulation(self):
        """
        Simulates the entire training lifecycle:
        Starting from a frozen backbone and thawing every unit sequentially.
        """
        # 1. Setup a fresh, frozen model
        print("=====================================================")
        model = self._frozen_resnet50()

        # 2. Get the units (Stages) identified by the Resolution Probe
        # For ResNet50, these are ordered from Output (Stage 4) to Input (Stage 1)
        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))

        initial_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"\n[Simulation Start] Initial trainable (Head only): {initial_trainable:,}"
        )

        # 3. Step through and thaw each unit
        for i, unit in enumerate(units):
            thaw_units(model, [unit])
            current_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # We expect the number of trainable parameters to increase at every step
            print(
                f"Step {i+1}: Unfroze Unit at {unit.resolution}. Trainable now: {current_trainable:,}"
            )

        final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(
            f"[Simulation End] Final trainable: {final_trainable:,} / Total: {total_params:,}"
        )

        # Validation: At the end, everything should be trainable
        self.assertEqual(final_trainable, total_params)


# ---------------------------------------------------------------------------
# 4. DynamicThawController
# ---------------------------------------------------------------------------


class TestDynamicThawController(unittest.TestCase):

    def _make_controller(self, n_units=4, patience=3, size=1, decay=0.1, base_lr=1e-3):
        units = _make_fake_units(n_units)
        return DynamicThawController(
            units=units,
            unfreeze_patience=patience,
            unfreeze_size=size,
            lr_decay_ratio=decay,
            base_lr=base_lr,
        )

    # --- no trigger while improving ---

    def test_no_trigger_on_first_step(self):
        ctrl = self._make_controller()
        result = ctrl.step(1.0)
        self.assertIsNone(result)

    def test_no_trigger_while_improving(self):
        ctrl = self._make_controller(patience=2)
        losses = [1.0, 0.9, 0.8, 0.7]
        for loss in losses:
            result = ctrl.step(loss)
            self.assertIsNone(result, f"Unexpected trigger at loss={loss}")

    def test_no_trigger_before_patience_reached(self):
        ctrl = self._make_controller(patience=3)
        ctrl.step(1.0)  # sets best
        # 2 non-improving steps — counter reaches 2, patience=3, no trigger
        self.assertIsNone(ctrl.step(1.0))
        self.assertIsNone(ctrl.step(1.0))

    # --- trigger at exactly patience ---

    def test_trigger_after_patience_epochs(self):
        ctrl = self._make_controller(patience=3)
        ctrl.step(1.0)  # epoch 1: set best
        ctrl.step(1.0)  # epoch 2: counter=1
        ctrl.step(1.0)  # epoch 3: counter=2
        result = ctrl.step(1.0)  # epoch 4: counter=3 >= patience → trigger
        self.assertIsNotNone(result)

    def test_trigger_returns_tuple(self):
        ctrl = self._make_controller(patience=2)
        ctrl.step(1.0)
        ctrl.step(1.0)
        result = ctrl.step(1.0)
        self.assertIsNotNone(result)
        units_batch, unit_lr = result
        self.assertIsInstance(units_batch, list)
        self.assertIsInstance(unit_lr, float)

    def test_trigger_returns_correct_number_of_units(self):
        ctrl = self._make_controller(n_units=4, patience=2, size=2)
        ctrl.step(1.0)
        ctrl.step(1.0)
        units_batch, _ = ctrl.step(1.0)
        self.assertEqual(len(units_batch), 2)

    def test_trigger_returns_correct_units_in_order(self):
        """First trigger should return the first `size` units from the list."""
        units = _make_fake_units(4)
        ctrl = DynamicThawController(
            units,
            unfreeze_patience=2,
            unfreeze_size=2,
            lr_decay_ratio=0.1,
            base_lr=1e-3,
        )
        ctrl.step(1.0)
        ctrl.step(1.0)
        units_batch, _ = ctrl.step(1.0)
        self.assertEqual(units_batch[0].stage_id, units[0].stage_id)
        self.assertEqual(units_batch[1].stage_id, units[1].stage_id)

    # --- counter resets after trigger ---

    def test_counter_resets_after_trigger(self):
        # Use patience=3 so patience-1=2 safe steps are verifiable after trigger
        ctrl = self._make_controller(n_units=4, patience=3)
        ctrl.step(1.0)  # step 1: sets best, counter=0
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)  # steps 2-4: counter 1→2→3 → TRIGGER
        # next patience-1=2 steps should not trigger again
        self.assertIsNone(ctrl.step(1.0))  # counter=1
        self.assertIsNone(ctrl.step(1.0))  # counter=2

    def test_second_trigger_returns_next_batch(self):
        units = _make_fake_units(4)
        ctrl = DynamicThawController(
            units,
            unfreeze_patience=2,
            unfreeze_size=1,
            lr_decay_ratio=0.1,
            base_lr=1e-3,
        )
        # First trigger fires at step 3 (best set at 1, then counter 1→2)
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)
        # Second trigger fires at step 5 (counter resets to 0 after first, then 1→2)
        ctrl.step(1.0)  # counter=1, no trigger
        result = ctrl.step(1.0)  # counter=2 → TRIGGER
        self.assertIsNotNone(result)
        units_batch, _ = result
        self.assertEqual(units_batch[0].stage_id, units[1].stage_id)

    # --- LR decay ---

    def test_first_trigger_lr(self):
        ctrl = self._make_controller(patience=2, decay=0.1, base_lr=1e-3)
        ctrl.step(1.0)
        ctrl.step(1.0)
        _, lr = ctrl.step(1.0)
        expected = 1e-3 * (0.1**1)
        self.assertAlmostEqual(lr, expected, places=10)

    def test_second_trigger_lr(self):
        ctrl = self._make_controller(n_units=4, patience=2, decay=0.1, base_lr=1e-3)
        # First trigger fires at step 3
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)
        # Second trigger fires at step 5
        ctrl.step(1.0)  # counter=1, no trigger
        _, lr = ctrl.step(1.0)  # counter=2 → TRIGGER
        expected = 1e-3 * (0.1**2)
        self.assertAlmostEqual(lr, expected, places=10)

    # --- exhaustion ---

    def test_all_unfrozen_flag(self):
        ctrl = self._make_controller(n_units=1, patience=2, size=1)
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)  # triggers, exhausts units
        self.assertTrue(ctrl.all_unfrozen)

    def test_no_trigger_when_exhausted(self):
        ctrl = self._make_controller(n_units=1, patience=2, size=1)
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)  # first (and only) trigger
        # Subsequent plateau should return None — no units left
        ctrl.step(1.0)
        ctrl.step(1.0)
        result = ctrl.step(1.0)
        self.assertIsNone(result)

    def test_improvement_after_trigger_delays_next(self):
        """After a trigger, a period of improvement should reset the counter."""
        ctrl = self._make_controller(n_units=4, patience=2)
        # First trigger
        ctrl.step(1.0)
        ctrl.step(1.0)
        ctrl.step(1.0)
        # Now loss improves — resets counter
        ctrl.step(0.5)
        # Only 1 non-improving step — should not trigger
        self.assertIsNone(ctrl.step(0.5))

    def test_size_clamped_at_remaining_units(self):
        """If fewer units remain than size, return only what's left."""
        ctrl = self._make_controller(n_units=2, patience=2, size=5)
        ctrl.step(1.0)
        ctrl.step(1.0)
        units_batch, _ = ctrl.step(1.0)
        self.assertEqual(len(units_batch), 2)  # only 2 available


# ---------------------------------------------------------------------------
# 5. create_optimizer — dynamic_unfreeze_mode
# ---------------------------------------------------------------------------


class TestCreateOptimizerDynamicMode(unittest.TestCase):

    def _make_options(self, lr=1e-3, optimizer="adamw"):
        return {
            "training": {
                "learning_rate": lr,
                "optimizer": optimizer,
                "weight_decay": 0.01,
                "dynamic_unfreeze": {
                    "unfreeze_size": 1,
                    "unfreeze_patience": 5,
                    "lr_decay_ratio": 0.1,
                },
            }
        }

    def _make_frozen_resnet50(self):
        options = {
            "model": {
                "backbone": "resnet50",
                "pretrained": False,
                "freeze_backbone": True,
                "classifier_hidden": [],
                "dropout": 0.0,
            }
        }
        return load_model(options, num_classes=4)

    def test_dynamic_mode_has_one_param_group(self):
        model = self._make_frozen_resnet50()
        options = self._make_options()
        optimizer = create_optimizer(
            model, options, "resnet50", dynamic_unfreeze_mode=True
        )
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_dynamic_mode_group_named_classifier(self):
        model = self._make_frozen_resnet50()
        options = self._make_options()
        optimizer = create_optimizer(
            model, options, "resnet50", dynamic_unfreeze_mode=True
        )
        self.assertEqual(optimizer.param_groups[0]["name"], "classifier")

    def test_dynamic_mode_only_classifier_params(self):
        model = self._make_frozen_resnet50()
        options = self._make_options()
        optimizer = create_optimizer(
            model, options, "resnet50", dynamic_unfreeze_mode=True
        )

        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        classifier_param_ids = {id(p) for p in model.fc.parameters()}

        # Optimizer should contain exactly the classifier params
        self.assertEqual(opt_param_ids, classifier_param_ids)

    def test_dynamic_mode_lr_matches_config(self):
        model = self._make_frozen_resnet50()
        options = self._make_options(lr=5e-4)
        optimizer = create_optimizer(
            model, options, "resnet50", dynamic_unfreeze_mode=True
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_normal_mode_unchanged(self):
        """Without dynamic_unfreeze_mode, behavior should be the same as before."""
        model = self._make_frozen_resnet50()
        options = {
            "training": {
                "learning_rate": 1e-3,
                "optimizer": "adamw",
                "weight_decay": 0.01,
            }
        }
        optimizer = create_optimizer(model, options, backbone_name=None)
        # Single param group containing only trainable (classifier) params
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_add_param_group_simulates_unfreeze(self):
        """
        Simulate what train_model does: start with classifier-only optimizer,
        then add a backbone unit's params after the first plateau trigger.
        """
        model = self._make_frozen_resnet50()
        options = self._make_options()
        optimizer = create_optimizer(
            model, options, "resnet50", dynamic_unfreeze_mode=True
        )

        units = get_unfreeze_units(model, "resnet50", (3, 224, 224))
        thaw_units(model, [units[0]])
        new_params = [
            p
            for name in units[0].module_names
            for p in model.get_submodule(name).parameters()
        ]
        optimizer.add_param_group(
            {"params": new_params, "lr": 1e-4, "name": "backbone_d1"}
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[1]["name"], "backbone_d1")
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
