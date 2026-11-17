# Shared energy data downloads

Copied from the user's supplied downloads on 2026-11-18. These are the original files; no normalization or rewriting was applied. This repository is public, so anyone can access these files.

## Files

| File | Format / encoding | Observed coverage | Records / contents | Source |
|---|---|---|---|---|
| `한국전력거래소_지역별 시간별 태양광 계량데이터(ESS 충방전량 제외)_20251231.zip` | ZIP containing 13 yearly CSV files | 2013–2025, one CSV per year | Archive integrity check passed | [KPX file page 15178526](https://www.data.go.kr/data/15178526/fileData.do) |
| `한국전력거래소_연료원별 전력거래량_20231231.csv` | CSV, CP949 | 2023-11-01–2023-12-31 | 30,744 data rows; this downloaded file does not contain January–October 2023 | [KPX file page 15081098](https://www.data.go.kr/data/15081098/fileData.do) |
| `한국지역난방공사_지역별 시간대별 계통한계가격 정보_20241231.csv` | CSV, UTF-8 with BOM | 2024-01-01–2024-12-31 | 26,352 data rows; categories: 육지, 제주, 육지+제주 | Provider page URL was not included with the downloaded file |

The district-heating SMP CSV header does not state a unit. Confirm the official source metadata before using its values in a model.

## SHA-256

- Solar ZIP: `679cc0c4d02980f5368c73221c0c1a6c236051e40e78fc4ad4f96b6c423cef73`
- Fuel CSV: `984b0064e098423eb181f854a157b4d45a0452bff0c653e8a28adb7ad29f9edc`
- District-heating SMP CSV: `b0ac899e25acad0d0a67ed3e38917f303662fb28bd90832331e1140d3aed1a08`
