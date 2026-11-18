# Issuer, security and business continuity

A current CIK identifies a filing entity; it does not by itself establish that a vendor ticker's entire backfilled price series represents the same security or business. A renamed holding company, a distribution, a new security and a simple symbol change require different treatment. The reference-date roster remains fixed; no issuer is replaced according to forecast performance.

## Documented conservative restrictions

| Reference ticker | Current spell starts | Evidence and reason |
|---|---|---|
| MTCH | 2020-07-01 | [SEC closing 8-K](https://www.sec.gov/Archives/edgar/data/891103/000110465920080606/tm206790d25_8k.htm), [Nasdaq action](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2020-109): old IAC/old Match separation, suspended old securities and new listings. CIK 891103 previously represented IAC. Cached prices start in 2014, before former Match's 2015 IPO; their predecessor linkage is not independently reconciled. |
| DD | 2019-06-03 | [Issuer announcement filed with SEC](https://www.sec.gov/Archives/edgar/data/1666700/000119312519163322/d715311dex991.htm): distributions from DowDuPont and current standalone DuPont trading. Earlier combined/predecessor histories are not automatically assigned to this business spell. |
| SW | 2024-07-08 | [Issuer listing announcement](https://www.smurfitkappa.com/us/newsroom/2024/smurfit-westrock-makes-its-debut-in-new-york-and-london): new Smurfit Westrock after the July 2024 combination. Cached vendor history predates this security; WestRock and Smurfit Kappa are not silently stitched. |
| KHC | 2015-07-06 | [Closing announcement filed with SEC](https://www.sec.gov/Archives/edgar/data/1545158/000119312515244355/d36612dex991.htm): KRFT ceased trading and KHC commenced. The existing price cache already begins at this date. |

These restrictions define a conservative sensitivity and the subsequent multimodal data contract. They do not establish that every earlier vendor return is numerically wrong, nor certify all other corporate histories. Ordinary mergers and name changes do not automatically invalidate all earlier observations; the listed cases have documented current-spell boundaries and unresolved predecessor linkage. The registry is curated and incomplete.

## Feature protection

`config/entity-spells-v1.json` records dates, reasons and primary sources. Earlier data and experiments remain immutable. The new `financial-entity-v1` dataset does the following:

- Excludes unverified predecessor origins from model fitting/evaluation without deleting cached source rows or replacing the issuer.
- Protects derived feature history as well as sequence endpoints: sixty input dates containing a sixty-return feature can depend on a price 119 offsets earlier; 512 trailing-rv20 inputs can depend on a price 531 offsets earlier. When an earlier vendor spell exists, require those complete post-boundary spans. An IPO-only cache without predecessor observations retains the original warm-up rule.
- Rebuilds affected financial snapshots from disclosures filed during the current spell. Later-filed comparative periods remain usable once disclosed; pre-spell filings cannot supply current-business values or rolling-flow components.
- Separates text boundary quality from identity eligibility. A correctly extracted 2017 IAC MD&A remains a valid document but is not a current-Match input under this contract. Unused frozen vectors may remain in local storage; they cannot enter training-document PCA or feature joins.

The SW restriction changes common-512 evaluation availability: it cannot supply 512 post-listing rv20 observations by the end of 2025. Do not describe differences in absolute losses between 81-firm and 80-firm evaluations as treatment effects. Compare forecasts on common keys when assessing changes.

## Interpretation

The original studies describe the declared reference/predecessor series and their limitations. Current-business conclusions remain developmental until the documented restrictions and their sensitivity are reported. This is an entity-data issue, distinct from directly using future target labels; no 2026 market outcomes were downloaded or scored. Source-value checks and date tests alone do not certify economic continuity.
