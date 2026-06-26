"""Auto-detect speakers and cameras so the GUI doesn't make Kevin type URLs.

- Speakers: ``pychromecast`` discovers Cast devices and returns their friendly
  names.
- Cameras: ONVIF WS-Discovery probes the LAN for cameras and resolves each
  one's RTSP stream URL.

Both functions degrade gracefully: on any error (library missing, nothing
found, network blocked) they return an empty list, and the GUI falls back to
manual entry with help text.
"""

from __future__ import annotations


def discover_speakers(timeout: float = 5.0) -> list[dict]:
    """Return Cast devices as ``[{"name", "model", "is_group"}]``.

    ``is_group`` flags multi-room speaker groups (casting to one plays on every
    member), so the GUI can warn about whole-house interruption.
    """
    try:
        import pychromecast
    except Exception:
        return []

    try:
        chromecasts, browser = pychromecast.get_chromecasts(timeout=timeout)
    except Exception:
        return []

    speakers = []
    try:
        for cc in chromecasts:
            info = cc.cast_info
            cast_type = getattr(info, "cast_type", "") or ""
            speakers.append(
                {
                    "name": info.friendly_name,
                    "model": getattr(info, "model_name", "") or "",
                    "is_group": cast_type == "group",
                }
            )
    finally:
        try:
            pychromecast.discovery.stop_discovery(browser)
        except Exception:
            pass
    # Stable, de-duplicated ordering for a tidy dropdown.
    seen, unique = set(), []
    for s in sorted(speakers, key=lambda d: d["name"].lower()):
        if s["name"] not in seen:
            seen.add(s["name"])
            unique.append(s)
    return unique


def discover_cameras(timeout: float = 4.0) -> list[dict]:
    """Return ONVIF cameras as ``[{"name", "rtsp_url", "host"}]``.

    Uses WS-Discovery to find cameras, then ONVIF media service to resolve each
    one's RTSP stream URI. Cameras that need credentials may not resolve a URL
    here; those come back with an empty ``rtsp_url`` and the GUI prompts for
    username/password + manual URL.
    """
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
    except Exception:
        return []

    hosts = _probe_onvif_hosts(WSDiscovery, timeout)
    cameras = []
    for host in hosts:
        name, rtsp = _resolve_onvif_camera(host)
        cameras.append({"name": name or host, "rtsp_url": rtsp, "host": host})
    return cameras


def _probe_onvif_hosts(WSDiscovery, timeout: float) -> list[str]:
    """WS-Discovery probe → unique camera host IPs."""
    import re

    wsd = WSDiscovery()
    hosts: list[str] = []
    try:
        wsd.start()
        services = wsd.searchServices(timeout=int(timeout))
        for service in services:
            for addr in service.getXAddrs():
                m = re.search(r"https?://([^/:]+)", addr)
                if m and m.group(1) not in hosts:
                    hosts.append(m.group(1))
    except Exception:
        return hosts
    finally:
        try:
            wsd.stop()
        except Exception:
            pass
    return hosts


def _resolve_onvif_camera(host: str) -> tuple[str, str]:
    """Best-effort ONVIF query for a camera's display name + RTSP URL.

    Returns ``(name, rtsp_url)``; either may be empty if ONVIF auth is required
    or the library is unavailable. Tries anonymous access only.
    """
    try:
        from onvif import ONVIFCamera
    except Exception:
        return "", ""

    for port in (80, 8000):
        try:
            cam = ONVIFCamera(host, port, "", "")
            media = cam.create_media_service()
            profiles = media.GetProfiles()
            if not profiles:
                continue
            token = profiles[0].token
            req = media.create_type("GetStreamUri")
            req.ProfileToken = token
            req.StreamSetup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri = media.GetStreamUri(req)
            name = ""
            try:
                name = cam.devicemgmt.GetDeviceInformation().Model or ""
            except Exception:
                pass
            return name, getattr(uri, "Uri", "") or ""
        except Exception:
            continue
    return "", ""


def probe_local_cameras(max_index: int = 5) -> list[dict]:
    """Find USB/built-in cameras on **this** machine (the one running the app).

    Returns ``[{"value": "usb:0", "label": "USB camera 0"}, …]`` for each index
    that opens and yields a frame. Empty on a headless box (e.g. a NAS). Stops
    early after two consecutive misses so it doesn't probe phantom devices.
    """
    from .detector import _open_capture

    found, misses = [], 0
    for i in range(max(0, max_index)):
        cap = _open_capture(f"usb:{i}")
        ok = False
        try:
            if cap.isOpened():
                ok, _ = cap.read()
        except Exception:
            ok = False
        finally:
            cap.release()
        if ok:
            found.append({"value": f"usb:{i}", "label": f"USB camera {i}"})
            misses = 0
        else:
            misses += 1
            if misses >= 2:
                break
    return found
