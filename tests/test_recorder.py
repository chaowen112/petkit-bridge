import unittest
from urllib.parse import parse_qs, urlsplit

import recorder


class CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            {"id": "11", "name": "solo", "camera": True},
            {"id": "22", "name": "fountain", "camera": False},
        ]
        self.states = {
            "11": {"device_type": "d4h"},
            "22": {"device_type": "ctw3"},
        }

    def test_selects_the_single_yumshare_solo_camera(self):
        self.assertEqual(recorder.select_d4h_device(self.devices, self.states), "11")

    def test_requires_explicit_id_when_multiple_solo_cameras_exist(self):
        devices = self.devices + [{"id": "33", "name": "second solo", "camera": True}]
        states = dict(self.states, **{"33": {"device_type": "d4h"}})
        with self.assertRaisesRegex(ValueError, "PETKIT_DEVICE_ID"):
            recorder.select_d4h_device(devices, states)
        self.assertEqual(recorder.select_d4h_device(devices, states, "33"), "33")

    def test_rejects_configured_id_that_is_not_a_d4h_camera(self):
        with self.assertRaisesRegex(ValueError, "not a YumShare Solo camera"):
            recorder.select_d4h_device(self.devices, self.states, "22")

    def test_reports_no_compatible_camera_without_device_details(self):
        with self.assertRaisesRegex(ValueError, "No YumShare Solo camera found"):
            recorder.select_d4h_device(self.devices[1:], self.states)


class StreamAndRecordingTests(unittest.TestCase):
    def test_whep_source_encodes_token_and_targets_device_endpoint(self):
        source = recorder.build_whep_source(
            "http://127.0.0.1:8787", "11", "a/b?c"
        )
        self.assertEqual(
            source,
            "webrtc:http://127.0.0.1:8787/device/11/whep?token=a%2Fb%3Fc#format=whep",
        )

    def test_go2rtc_registration_url_preserves_nested_whep_url(self):
        source = "webrtc:http://127.0.0.1:8787/device/11/whep?token=a%2Fb#format=whep"
        url = recorder.go2rtc_registration_url(
            "http://127.0.0.1:1984", "petkit_solo", source
        )
        parsed = urlsplit(url)
        self.assertEqual(parsed.path, "/api/streams")
        self.assertEqual(
            parse_qs(parsed.query), {"name": ["petkit_solo"], "src": [source]}
        )

    def test_ffmpeg_records_copy_only_video_in_recoverable_segments(self):
        command = recorder.build_ffmpeg_command(
            "rtsp://127.0.0.1:8554/petkit_solo",
            "/recordings/yumshare-solo-%Y%m%d-%H%M%S.mkv",
            300,
            record_audio=False,
        )
        self.assertIn("-n", command)
        self.assertIn("-c", command)
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertEqual(command[command.index("-segment_time") + 1], "300")
        self.assertIn("-segment_format", command)
        self.assertEqual(command[command.index("-segment_format") + 1], "matroska")
        self.assertIn("-map", command)
        self.assertIn("0:v:0", command)
        self.assertNotIn("0:a?", command)

    def test_ffmpeg_includes_audio_only_when_explicitly_enabled(self):
        command = recorder.build_ffmpeg_command(
            "rtsp://127.0.0.1:8554/petkit_solo",
            "/recordings/yumshare-solo-%Y%m%d-%H%M%S.mkv",
            300,
            record_audio=True,
        )
        self.assertIn("0:a?", command)


class RetentionTests(unittest.TestCase):
    def test_retention_defaults_to_thirty_days(self):
        self.assertEqual(recorder.parse_retention_days(None), 30)
        self.assertEqual(recorder.parse_retention_days(""), 30)

    def test_retention_accepts_one_through_thirty_days(self):
        self.assertEqual(recorder.parse_retention_days("1"), 1)
        self.assertEqual(recorder.parse_retention_days("30"), 30)

    def test_retention_rejects_zero_and_values_over_thirty_days(self):
        for value in ("0", "31", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "1 to 30"):
                    recorder.parse_retention_days(value)

    def test_zero_retention_is_not_allowed_to_disable_cleanup(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            old = Path(directory) / "yumshare-solo-old.mkv"
            old.write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "1 to 30"):
                recorder.cleanup_old_recordings(directory, 0, now=10_000_000)
            self.assertTrue(old.exists())

    def test_positive_retention_removes_only_expired_mkv_files(self):
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            old = Path(directory) / "yumshare-solo-old.mkv"
            recent = Path(directory) / "yumshare-solo-recent.mkv"
            unrelated = Path(directory) / "other-camera.mkv"
            other = Path(directory) / "notes.txt"
            old.write_bytes(b"old")
            recent.write_bytes(b"new")
            unrelated.write_bytes(b"leave this alone")
            other.write_text("keep", encoding="utf-8")
            now = 10_000_000
            os.utime(old, (now - 3 * 86400, now - 3 * 86400))
            os.utime(recent, (now - 60, now - 60))
            removed = recorder.cleanup_old_recordings(directory, 1, now=now)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(recent.exists())
            self.assertTrue(unrelated.exists())
            self.assertTrue(other.exists())


if __name__ == "__main__":
    unittest.main()
