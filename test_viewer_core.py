from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import view_bentron_aoi_3d_gl as viewer


SAMPLES_DIR = Path(__file__).with_name("samples")


def sample_path(name: str) -> Path:
    return SAMPLES_DIR / f"{name}.ptt"


class ViewerCoreTests(unittest.TestCase):
    def require_sample(self, name: str) -> Path:
        path = sample_path(name)
        if not path.exists():
            self.skipTest(f"sample not present: {path}")
        return path

    def test_read_ptt_layout(self) -> None:
        path = self.require_sample("1@206")
        width, height, pitch_x, pitch_y, planes = viewer.read_ptt(path)
        self.assertEqual((width, height), (524, 239))
        self.assertEqual(planes.shape, (3, height, width))
        self.assertGreater(pitch_x, 0.0)
        self.assertGreater(pitch_y, 0.0)

    def test_read_pot_layout(self) -> None:
        self.require_sample("1@206")
        pot = viewer.read_pot(SAMPLES_DIR / "1@206.pot")
        self.assertIsNotNone(pot)
        assert pot is not None
        width, height, planes = pot
        self.assertEqual((width, height), (524, 239))
        self.assertEqual(planes.shape, (5, height, width))

    def test_flipx_does_not_change_height_distribution(self) -> None:
        path = self.require_sample("1@206")
        sample = viewer.find_samples([path])[0]
        normal = viewer.build_mesh(sample, grid=180, visual_z=0.65, flip_x=False)
        flipped = viewer.build_mesh(sample, grid=180, visual_z=0.65, flip_x=True)
        np.testing.assert_allclose(np.sort(normal.vertices[:, 2]), np.sort(flipped.vertices[:, 2]))

    def test_debug_textures_have_rgb_output(self) -> None:
        path = self.require_sample("1@836")
        sample = viewer.find_samples([path])[0]
        texture = viewer.choose_texture(sample, use_ac=False)
        for mode in viewer.DEBUG_TEXTURE_MODES[1:]:
            image = viewer.make_debug_texture(sample, texture, mode)
            self.assertEqual(image.mode, "RGB")
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)


if __name__ == "__main__":
    unittest.main()

