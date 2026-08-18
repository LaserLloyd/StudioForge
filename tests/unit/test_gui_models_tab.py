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
        assert tab._GRID_TEMPLATE.split() == [
            f"var(--sfm-{key})" for key, _ in tab._DEFAULT_WIDTHS
        ]
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
