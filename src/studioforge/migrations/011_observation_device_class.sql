-- 011_observation_device_class.sql: what kind of cards, which split, which file (D88).
--
-- D51 reads ONE observation row back at plan time and spends its measured
-- footprint as the estimate, so every knob that moves the footprint has to be
-- in the match key. Until now the key named the device COUNT only. A
-- correction learned on a pair of 32 GiB Blackwell cards was therefore spent
-- on a pair of 24 GiB Ampere cards for the same model id; a layer split and a
-- tensor split of the same model shared one row; and a GGUF replaced under the
-- same id inherited the old file's correction. Three columns close that:
--
-- * class_key  -- the placement's card classes, the portable grouping key
--   "<arch>-<GiB>g" per card (sm_86-24g for a 24 GiB Ampere card), sorted and
--   joined with "+": "sm_86-24g+sm_86-24g" for that pair. Never a product
--   name, never a CUDA index (D42 moves models between same-class sets).
-- * split_mode -- how the cards share the model: "none" on one card, else
--   the mode the child was launched with ("layer", "row", "tensor").
-- * gguf_size  -- the model file's size in bytes, so a replaced file under the
--   same id starts its own history.
--
-- Rows written before this migration keep NULL in all three and stay usable
-- as LEGACY matches (owner decision F-6): Database.matching_observation still
-- returns such a row when no row of the exact class/split/size exists, and
-- always prefers a row that names all three. A non-NULL value never matches a
-- different one.
ALTER TABLE load_observations ADD COLUMN class_key TEXT;
ALTER TABLE load_observations ADD COLUMN split_mode TEXT;
ALTER TABLE load_observations ADD COLUMN gguf_size INTEGER;
