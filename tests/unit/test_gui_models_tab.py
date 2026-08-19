"""Layout invariants of the Models tab table."""

from __future__ import annotations


class TestResizableColumns:
    """The Models table is a CSS grid whose track widths are user-draggable."""

    def test_every_sortable_column_has_a_width_variable(self) -> None:
        from studioforge.gui import state as st
        from studioforge.gui.tabs import models as tab

        # A column added to MODEL_COLUMNS without a matching grid track would
        # render into whatever track happened to be next, silently shifting
        # every cell to its right by one column.
        assert set(tab._COLUMN_VARS) == {c.key for c in st.MODEL_COLUMNS}
        widths = {key for key, _ in tab._DEFAULT_WIDTHS}
        assert set(tab._COLUMN_VARS.values()) <= widths

    def test_grid_template_covers_every_track_including_actions(self) -> None:
        from studioforge.gui.tabs import models as tab

        assert "actions" in {key for key, _ in tab._DEFAULT_WIDTHS}
        # One track per column, in declaration order.
        assert tab._GRID_TEMPLATE.split() == [f"var(--sfm-{key})" for key, _ in tab._DEFAULT_WIDTHS]
        for key, _ in tab._DEFAULT_WIDTHS:
            assert f"--sfm-{key}:" in tab._ROOT_VARS

    def test_cells_clip_instead_of_overflowing(self) -> None:
        from studioforge.gui.tabs import models as tab

        # The bug this replaced: flex items default to min-width:auto, so a
        # long model id pushed every column to its right off the screen.
        assert "min-width:0" in tab._TABLE_CSS
        assert "text-overflow:ellipsis" in tab._TABLE_CSS
        assert ".sfm-row>*{min-width:0;overflow:hidden;}" in tab._TABLE_CSS
        # Header cells must not clip, or the drag handle would be invisible.
        assert ".sfm-head>*{overflow:visible;position:relative;}" in tab._TABLE_CSS

    def test_resize_script_is_idempotent_and_persists(self) -> None:
        from studioforge.gui.tabs import models as tab

        # Installed per page render; must not stack listeners on the document.
        assert "__sfmResizeReady" in tab._TABLE_JS
        assert "localStorage" in tab._TABLE_JS
        # Widths go on :root so a table.refresh() cannot wipe them.
        assert "document.documentElement" in tab._TABLE_JS


class TestOptimalSettingsBlock:
    """The "Optimal settings" lines the settings dialog shows (D36).

    Pure formatters, tested without a browser: the Models tab is the one place a
    person compares "what would this do on the 3090s" against "what would it do
    on everything", and a line that reads badly is the whole feature lost.
    """

    @staticmethod
    def _profiles() -> dict:
        return {
            "model_id": "pub/gemma-31b",
            "modes": [
                {
                    "mode": "dual_5090",
                    "label": "2x RTX 5090",
                    "devices": [0, 1],
                    "fits_now": True,
                    "would_evict": [],
                    "optimal": {
                        "ctx_per_slot": 131072,
                        "kv_cache_type": "f16",
                        "kv_cache_type_v": "f16",
                        "max_parallel": 3,
                        "est_gen_tps": 48.2,
                        "est_gen_tps_full_ctx": 41.0,
                        "load_args": {
                            "model_id": "pub/gemma-31b",
                            "ctx_size": 131072,
                            "parallel": 3,
                            "kv_cache_type": "f16",
                            "devices": [0, 1],
                        },
                    },
                },
                {
                    "mode": "dual_3090",
                    "label": "2x RTX 3090",
                    "devices": [2, 3],
                    "fits_now": False,
                    "would_evict": ["pub/other", "pub/another"],
                    "optimal": {
                        "ctx_per_slot": 65536,
                        "kv_cache_type": "q8_0",
                        "kv_cache_type_v": "q4_0",
                        "max_parallel": 1,
                        "est_gen_tps": 22.0,
                        "est_gen_tps_full_ctx": 18.0,
                        "load_args": {"model_id": "pub/gemma-31b", "devices": [2, 3]},
                    },
                },
                {
                    "mode": "single_5090",
                    "label": "1x RTX 5090",
                    "devices": [0],
                    "fits_now": False,
                    "would_evict": [],
                    "optimal": None,
                },
            ],
        }

    def test_a_line_reads_as_a_sentence(self) -> None:
        from studioforge.gui import state as st

        lines = st.placement_lines(self._profiles())
        assert lines[0].text == (
            "2x RTX 5090: 131072 ctx · f16 · 3 slots · ~48 t/s (~41 full) · fits now"
        )

    def test_an_asymmetric_kv_pair_is_spelled_out(self) -> None:
        from studioforge.gui import state as st

        assert "q8_0/q4_0" in st.placement_lines(self._profiles())[1].summary

    def test_what_stands_in_the_way_is_named_and_counted(self) -> None:
        from studioforge.gui import state as st

        line = st.placement_lines(self._profiles())[1]
        assert line.availability == "needs 2 unloads (pub/other, pub/another)"
        assert line.colour == "warning"

    def test_a_mode_too_small_says_so_rather_than_offering_a_load(self) -> None:
        from studioforge.gui import state as st

        line = st.placement_lines(self._profiles())[2]
        assert line.load_args == {}
        assert "too small" in line.availability
        assert line.colour == "negative"

    def test_the_load_args_are_carried_through_verbatim(self) -> None:
        from studioforge.gui import state as st

        line = st.placement_lines(self._profiles())[0]
        assert line.load_args["devices"] == [0, 1]
        assert line.load_args["ctx_size"] == 131072

    def test_the_headline_names_the_default_placement(self) -> None:
        from studioforge.gui import state as st

        assert st.placement_headline(self._profiles()).startswith("Recommended: 2x RTX 5090:")

    def test_a_model_that_fits_nowhere_says_so(self) -> None:
        from studioforge.gui import state as st

        empty = {"modes": [{"mode": "single_5090", "label": "1x", "optimal": None}]}
        assert "No hardware mode" in st.placement_headline(empty)

    def test_the_block_is_filled_lazily_so_the_dialog_opens_fast(self) -> None:
        import inspect

        from studioforge.gui.tabs import models as tab

        source = inspect.getsource(tab._settings_dialog)
        assert "ui.timer(" in source
        assert "_render_placements" in source
