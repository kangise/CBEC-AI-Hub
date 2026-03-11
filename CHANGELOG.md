# Changelog

All notable changes to CBEC-AI-Hub will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [v1.1.0] - 2025-06-20

### Added
- `notebooks/` directory with first Notebook: b1-data-pipeline.ipynb (Amazon 报告数据处理)
- `README_EN.md` — complete English version of README
- "Top 10 Prompts" viral entry section in README.md
- "What's New" section at top of README.md
- New Issue templates: broken link report, prompt submission, notebook submission
- `CODEOWNERS` file for automated review assignment
- `CHANGELOG.md` (this file)
- Case studies: AI-Powered Listing Generation, Automated Review Analysis
- Contributors section in README.md
- Updated PR template with quality checklist
- SEO configuration in `_config.yml`

### Changed
- README.md first screen redesigned with bilingual tagline, badges, Mermaid diagram, and "Try This Now" section
- Updated link-checker workflow to scan all Markdown directories
- Updated `CONTRIBUTING.md` with Prompt template submission guide
- Updated `paths/b-developers/b1-data-pipeline.md` with Open in Colab badge

## [v1.0.0] - 2025-06-15

### Added
- Modularized content: `paths/` directory with A1-A6, B1-B5, C1-C3 modules
- `prompts/` directory with 5 standardized Prompt template files
- `roadmap/` directory with public roadmap and coverage map
- Competitive landscape analysis in `docs/competitive-analysis.md`
- Content quality test infrastructure (`tests/test_repo_properties.py`)
- Fixed `_config.yml` metadata to match README content
- Jekyll SEO plugins (jekyll-seo-tag, jekyll-sitemap)

### Changed
- README.md slimmed down to navigation hub (< 500 lines)
- All broken internal links fixed or removed

### Fixed
- Removed all "即将发布" placeholders without tracking Issues
- Fixed _config.yml title and description mismatch
