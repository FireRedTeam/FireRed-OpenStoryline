__version__ = "0.1.0"

# Register the HEIF/HEIC opener with Pillow so that downstream nodes that read
# images via PIL.Image.open() (load_media, render_video, sampling_handler, ...)
# transparently support .heic / .heif files produced by iPhone cameras.
try:
    from pillow_heif import register_heif_opener as _register_heif_opener

    _register_heif_opener()
except ImportError:
    # pillow-heif is an optional dependency; if missing, .heic uploads will
    # still be filtered out by the suffix whitelist with a clear log line.
    pass
