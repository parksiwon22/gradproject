# SMP automated collection and issue timing

The forecast workflow keeps the research issue origin fixed at 23:00 Asia/Seoul. Its daily 22:45 run refreshes EPSIS near the origin and only admits rows whose recorded `first_seen` is no later than that origin. GitHub Actions schedule events are best-effort and have no guaranteed start time. A late run therefore cannot backfill yesterday's forecast from prices first collected after yesterday's 23:00 cutoff; it records a withheld result instead.

The `SMP durable pre-collection` workflow runs at 20:00, 21:00, and 22:00 KST (cron is UTC). It only collects and persists keyless EPSIS mainland SMP to the `smp-live-data` branch. It never creates forecasts. The pre-collection workflow and forecast workflow share a concurrency group while writing durable state. The forecast workflow still performs its own near-origin refresh, so pre-collection is an additional chance to have recent data available, not a guarantee that an origin-time observation exists.

The available run evidence is not enough to certify a better fixed issue time. Run `42897049481` began at 05:37 KST after its expected 22:45 start and was withheld against the original prior-day 23:00 origin because the origin-time SMP had no qualifying receipt. That run also cannot establish the normal EPSIS publication time: its collection occurred after the cutoff. A historical example showed next-day official price on the KPX public page before an issue check, but that probe establishes only what that page showed at that moment, not the first publication time across all official channels. D+1 remains a reference when its target price is already public; D+2/D+3 are eligible for prospective evaluation only after a successful pre-origin publication check.

The 23:00 research cutoff is fixed in the 275-day study. Current candidates also consume the 23:00 SMP point and preceding 168-hour window. Moving the live cutoff earlier would change available features and require a new as-of walk-forward evaluation; changing the clock solely to increase issuance counts would not validate the same model. The least disruptive proposal is to keep 23:00, collect repeatedly beforehand, and use persisted receipt timestamps to assess whether another cutoff has enough complete inputs before proposing a time-aligned study.

The automated D+2/D+3 forecast is issued only when all these checks pass:

- The fixed 23:00 origin has arrived.
- An SMP observation at the origin exists in the durable database and its `first_seen` is at or before 23:00. An observation collected afterward is excluded even if it carries an earlier market timestamp.
- The recent 168-hour input window is complete and the D+2/D+3 seasonal-anchor lags are usable.
- The relevant checkpoint exists and its training cutoff precedes the origin.
- A forecast for that origin has not already been issued. Retries retain the original record rather than appending a duplicate.

Insufficient input, collection/API failure, missing checkpoint/anchors, or an unelapsed origin results in a withheld status. A withheld run does not change its origin to the time at which a delayed runner happened to start. Official-price availability checks and later same-target-date comparisons remain separate from forecast generation.

Pre-collection schedules cannot guarantee a timely run or data publication: GitHub may delay or coalesce scheduled workflow events, and EPSIS may not yet have exposed the latest interval. Until this PR is merged and the schedule workflows run on the default branch, Actions on this PR validate the workflow and tests but do not prove live scheduled operation.
