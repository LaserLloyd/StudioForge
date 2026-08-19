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


class TestSlotsAndExactContext:
    """The WP19 half of the placements block: slot counts, and "load at 256k"."""

    @staticmethod
    def _entry(*, max_parallel: int, recommended: int | None, basis: str = "estimated") -> dict:
        optimal: dict = {"max_parallel": max_parallel}
        if recommended is not None:
            optimal["recommended_parallel"] = recommended
            optimal["recommended_parallel_basis"] = basis
        return {"optimal": optimal}

    def test_the_slot_line_stays_quiet_until_it_has_something_to_say(self) -> None:
        """Estimated == ceiling, so printing both would print one number twice."""
        from studioforge.gui import state as st

        assert st.parallel_summary(self._entry(max_parallel=4, recommended=4)) == ""
        assert st.parallel_summary(self._entry(max_parallel=4, recommended=None)) == ""
        assert st.parallel_summary({"optimal": None}) == ""

    def test_a_measurement_that_lowers_the_count_is_shown_with_its_basis(self) -> None:
        from studioforge.gui import state as st

        summary = st.parallel_summary(self._entry(max_parallel=7, recommended=2, basis="measured"))
        assert summary == "2 of 7 slots (measured)"

    def test_the_four_context_buttons_are_the_ones_the_mcp_tool_names(self) -> None:
        """A user asking for "256k" and an agent asking for 262144 mean one load."""
        from studioforge.gui import state as st

        assert [ctx for _label, ctx in st.CTX_BUTTONS] == [65536, 131072, 262144, 524288]

    def test_a_size_past_the_trained_window_is_greyed_and_says_why(self) -> None:
        from studioforge.gui import state as st

        profiles = {
            "n_ctx_train": 131072,
            "modes": [{"optimal": {"ctx_per_slot": 131072}}],
        }
        buttons = {b.label: b for b in st.ctx_buttons(profiles)}
        assert buttons["128k"].enabled is True
        assert buttons["256k"].enabled is False
        assert "trained to 131072" in buttons["256k"].tooltip

    def test_a_size_no_placement_reaches_is_greyed_for_a_different_reason(self) -> None:
        """One is permanent and about the model; the other may change on an unload."""
        from studioforge.gui import state as st

        profiles = {
            "n_ctx_train": 1048576,
            "modes": [{"optimal": {"ctx_per_slot": 65536}}],
        }
        buttons = {b.label: b for b in st.ctx_buttons(profiles)}
        assert buttons["64k"].enabled is True
        assert buttons["128k"].enabled is False
        assert "no set of cards" in buttons["128k"].tooltip

    def test_the_measured_curve_renders_as_an_aligned_table(self) -> None:
        from studioforge.gui import state as st

        report = {
            "levels": [
                {
                    "n_streams": 1,
                    "per_stream_tps": 48.0,
                    "aggregate_tps": 48.0,
                    "p95_latency_s": 2.7,
                    "achieved_batch": 1.0,
                },
                {
                    "n_streams": 2,
                    "per_stream_tps": 44.0,
                    "aggregate_tps": 88.0,
                    "p95_latency_s": 3.0,
                    "achieved_batch": 2.0,
                },
            ],
            "recommended_parallel_detail": "2 slots: aggregate 88.0 t/s",
        }
        lines = st.parallel_level_lines(report)
        assert lines[0].split() == ["N", "per-stream", "aggregate", "p95", "batch"]
        assert len(lines) == 3
        assert "2.0x" in lines[2]
        assert st.parallel_verdict(report).startswith("Recommended: 2 slots")

    def test_a_level_with_no_numbers_renders_as_a_dash_not_a_crash(self) -> None:
        from studioforge.gui import state as st

        lines = st.parallel_level_lines({"levels": [{"n_streams": 4, "error": "boom"}]})
        assert "-" in lines[1]

    def test_the_measure_button_uses_the_shared_runner(self) -> None:
        """A second ParallelBenchmarker would be a second lock, which is no lock."""
        import inspect

        from studioforge.gui.tabs import models as tab

        source = inspect.getsource(tab._measure_parallel)
        assert "parallel_bench.for_state(ctx.api_state)" in source

    def test_the_exact_context_buttons_call_load_recommended(self) -> None:
        import inspect

        from studioforge.gui.tabs import models as tab

        assert "load_recommended" in inspect.getsource(tab._load_at_ctx)
        assert "_render_ctx_buttons" in inspect.getsource(tab._render_placements)
