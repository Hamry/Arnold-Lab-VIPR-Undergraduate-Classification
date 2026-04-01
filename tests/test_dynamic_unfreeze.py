"""
Unit tests for the Dynamic Unfreezing system.

Covers:
    - get_unfreeze_units()        : registry-based block partitioning
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


def _make_fake_units(n):
    """Return n synthetic UnfreezeUnit objects."""
    return [
        UnfreezeUnit(
            stage_id=i, module_names=[f"layer{i}"], parameter_count=100 * (i + 1)
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. get_unfreeze_units
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
        units = get_unfreeze_units(model, "resnet50")
        self.assertIsInstance(units, list)
        for u in units:
            self.assertIsInstance(u, UnfreezeUnit)

    def test_units_not_empty(self):
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50")
        self.assertGreater(len(units), 0)

    def test_resnet50_has_expected_block_count(self):
        """ResNet50 registry defines 5 blocks: stem + layer1-4."""
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50")
        self.assertEqual(len(units), 5)

    def test_classifier_excluded_from_units(self):
        """No unit should contain the 'fc' classifier module."""
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50")
        all_names = [name for u in units for name in u.module_names]
        self.assertFalse(any(name == "fc" for name in all_names))

    def test_ordered_output_to_input(self):
        """For ResNet50, units[0] is layer4 and units[-1] is the stem."""
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50")

        # First unit (closest to head) should be layer4
        self.assertEqual(
            units[0].module_names,
            ["layer4"],
            f"Expected units[0] to be ['layer4'], got {units[0].module_names}",
        )

        # Last unit (furthest from head) should be the stem [conv1, bn1]
        self.assertEqual(
            units[-1].module_names,
            ["conv1", "bn1"],
            f"Expected units[-1] to be ['conv1', 'bn1'], got {units[-1].module_names}",
        )

    def test_parameter_counts_positive(self):
        model = self._make_resnet50()
        units = get_unfreeze_units(model, "resnet50")
        for u in units:
            self.assertGreater(u.parameter_count, 0)

    def test_unknown_backbone_raises(self):
        model = self._make_resnet50()
        with self.assertRaises(ValueError):
            get_unfreeze_units(model, "nonexistent_backbone_xyz")

    def test_vit_returns_12_units(self):
        """ViT-B/16 registry has 12 transformer blocks."""
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16")
        self.assertEqual(len(units), 12)

    def test_vit_units_are_transformer_blocks(self):
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16")
        # All module_names should be of the form "encoder.layers.encoder_layer_N"
        for u in units:
            for name in u.module_names:
                self.assertTrue(
                    name.startswith("encoder.layers.encoder_layer_"),
                    f"Expected encoder.layers.encoder_layer_N, got {name!r}",
                )

    def test_vit_ordered_output_to_input(self):
        """First unit should be block 11 (last = closest to head); last is block 0."""
        vit = models.vit_b_16(weights=None)
        vit.heads = nn.Linear(768, 4)
        units = get_unfreeze_units(vit, "vit_b_16")
        self.assertIn("encoder.layers.encoder_layer_11", units[0].module_names)
        self.assertIn("encoder.layers.encoder_layer_0", units[-1].module_names)


# ---------------------------------------------------------------------------
# 2. thaw_units
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
        units = get_unfreeze_units(model, "resnet50")

        before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        thaw_units(model, [units[0]])
        after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertGreater(after, before)

    def test_thaw_returns_correct_count(self):
        model = self._frozen_resnet50()
        units = get_unfreeze_units(model, "resnet50")

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
        units = get_unfreeze_units(model, "resnet50")

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
        units = get_unfreeze_units(model, "resnet50")

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
        units = get_unfreeze_units(model, "resnet50")

        # Thaw only the first unit
        thaw_units(model, [units[0]])

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
        print("=====================================================")
        model = self._frozen_resnet50()

        units = get_unfreeze_units(model, "resnet50")

        initial_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(
            f"\n[Simulation Start] Initial trainable (Head only): {initial_trainable:,}"
        )

        for i, unit in enumerate(units):
            thaw_units(model, [unit])
            current_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(
                f"Step {i+1}: Unfroze {unit.module_names}. Trainable now: {current_trainable:,}"
            )

        final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        print(
            f"[Simulation End] Final trainable: {final_trainable:,} / Total: {total_params:,}"
        )

        # At the end, everything should be trainable
        self.assertEqual(final_trainable, total_params)


# ---------------------------------------------------------------------------
# 3. DynamicThawController
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
# 4. create_optimizer — dynamic_unfreeze_mode
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

        units = get_unfreeze_units(model, "resnet50")
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
