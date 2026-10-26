-- Vectors already fetched from the embedding model, so re-indexing and eval reruns only
-- pay for texts not seen before. Keyed by model, since vectors from two models differ.
-- double precision, so a cached vector scores exactly like a fresh one.
CREATE TABLE IF NOT EXISTS embedding_cache (
    model       text   NOT NULL,
    text_sha256 text   NOT NULL,
    vector      double precision[] NOT NULL,
    PRIMARY KEY (model, text_sha256)
);
