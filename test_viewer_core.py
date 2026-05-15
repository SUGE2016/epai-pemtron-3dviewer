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

    def test_invalid_quad_culling_removes_faces_only(self) -> None:
        path = self.require_sample("1@206")
        sample = viewer.find_samples([path])[0]
        culled = viewer.build_mesh(sample, grid=180, visual_z=0.65, cull_invalid_quads=True)
        unculled = viewer.build_mesh(sample, grid=180, visual_z=0.65, cull_invalid_quads=False)
        self.assertEqual(culled.vertices.shape, unculled.vertices.shape)
        self.assertLessEqual(culled.indices.size, unculled.indices.size)

    def test_invalid_height_fill_returns_repaired_mask(self) -> None:
        z = np.zeros((5, 5), dtype=np.float32)
        z[2, 1] = 4.0
        z[1, 2] = 4.0
        z[2, 3] = 4.0
        z[3, 2] = 4.0
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 1] = mask[1, 2] = mask[2, 3] = mask[3, 2] = True
        filled, repaired = viewer.fill_invalid_height(z, mask, return_mask=True)
        self.assertTrue(repaired[2, 2])
        self.assertGreater(filled[2, 2], 0.0)

    def test_plane0_repair_fills_invalid_with_low_surface(self) -> None:
        planes = np.full((3, 9, 9), 100, dtype=np.uint16)
        planes[0, 4, 4] = 65535
        planes[1, 4, 4] = 20000
        planes[2, 4, 4] = 30000
        repaired, mask = viewer.repair_plane0_invalid_as_low_surface(planes)
        self.assertTrue(mask[4, 4])
        self.assertLess(repaired[4, 4], 1000)

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
