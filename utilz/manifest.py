import json
import warnings
from pathlib import Path


class Manifest:
    output_attr: str = "output_fldr"
    """Write selected owner attributes to a JSON manifest in its output folder."""

    format = "utilz_manifest_v1"

    def __init__(
        self,
        owner,
        keys: str,
        manifest_name: str = "manifest.json",
    ):
        """`keys` is a whitespace/comma separated list of owner attribute names."""
        self.keys = keys.replace(",", " ").split()
        self.manifest_name = manifest_name

    def payload(self) -> dict:  #AI
        """Create the JSON-compatible manifest payload."""
        return {
            "format": self.format,
            "owner": self.owner.__class__.__name__,
            "params": {
                key: self._json_value(getattr(self.owner, key)) for key in self.keys
            },
        }

    def write(self) -> Path:  #AI
        """Compare any existing manifest, then write the current payload atomically."""
        return self.write_payload(self.manifest_fn, self.payload(), compare_key="params")

    @classmethod
    def write_payload(
        cls, manifest_fn: Path, payload: dict, compare_key: str | None = None
    ) -> Path:  #AI
        """Write an existing manifest schema with atomic replacement."""
        manifest_fn = Path(manifest_fn)
        manifest_fn.parent.mkdir(parents=True, exist_ok=True)
        if manifest_fn.exists():
            cls.warn_mismatches(
                json.loads(manifest_fn.read_text()), payload, compare_key=compare_key
            )
        tmp_fn = manifest_fn.with_suffix(".json.tmp")
        tmp_fn.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_fn.replace(manifest_fn)
        return manifest_fn

    @classmethod
    def warn_mismatches(
        cls, old: dict, new: dict, compare_key: str | None = None
    ) -> None:  #AI
        """Warn for parameter values that differ from an existing manifest."""
        old_params = old[compare_key] if compare_key is not None else old
        new_params = new[compare_key] if compare_key is not None else new
        for key in sorted(new_params):
            old_value = old_params[key] if key in old_params else "<MISSING>"
            if old_value != new_params[key]:
                warnings.warn(
                    "Manifest mismatch for {0}: existing={1!r} current={2!r}".format(
                        key, old_value, new_params[key]
                    ),
                    stacklevel=2,
                )

    @property
    def manifest_fn(self) -> Path:
        output_fldr = Path(getattr(self.owner, self.output_attr))
        return output_fldr / self.manifest_name

    @classmethod
    def _json_value(cls, value):  #AI
        """Convert common Python/path containers to JSON-compatible values."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, tuple):
            return [cls._json_value(item) for item in value]
        if isinstance(value, list):
            return [cls._json_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._json_value(val) for key, val in value.items()}
        return str(value)
