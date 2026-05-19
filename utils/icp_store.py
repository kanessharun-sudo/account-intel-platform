"""Persist the user's ICP definition to a local JSON file.

The DEFAULT_ICP_TEMPLATE below ships the Ctruh ICP as the out-of-the-box
default, so users don't need to re-paste it on every visit. They can edit
it in the UI and click "Save ICP" to persist their changes, or click
"Reload saved" to revert to the last saved version. If no save has ever
happened, the default below is used.
"""
from __future__ import annotations

import json
from pathlib import Path

ICP_PATH = Path(__file__).resolve().parent.parent / "storage" / "icp.json"

DEFAULT_ICP_TEMPLATE = """\
## What my company does
Ctruh is a Bengaluru-based deep-tech company building browser-native, no-code 3D and XR infrastructure for businesses. Its flagship product, Commverse Studio, is an AI-powered Unified XR Commerce Studio that lets brands create 3D product viewers, AR try-ons, virtual stores, and AI-generated ad creatives with no code and no app installs. The underlying stack — a proprietary 3D engine plus VersaAI, a generative AI model that converts images, text, and video into production-ready 3D assets — is designed to let brands deploy immersive experiences in minutes rather than weeks.

## Target industries
- Retail, e-commerce, and D2C brands (primary focus — apparel, footwear, accessories, beauty, furniture)
- Real estate (virtual walkthroughs, property visualization)
- Automotive (product configurators, virtual showrooms)
- Education (immersive learning content)
- Healthcare (training and visualization use cases)

Note: Commverse Studio is most heavily aimed at retail/e-commerce/D2C. Real estate, automotive, education, and healthcare are stated verticals but currently served more through the underlying 3D engine than a dedicated product.

## Target company size
- D2C / e-commerce sweet spot: brands with ~50–1,000 SKUs and meaningful online GMV (Series A+ or established bootstrapped brands doing $1M+ ARR online)
- Mid-market to enterprise for real estate, automotive, and large retail (where cost-per-shoot and return-rate pain justify platform spend)
- Likely too small: pre-revenue brands or hobby sellers with <10 SKUs and no online store of their own

## Target geographies
- India (home market — Bengaluru HQ, existing brand customers)
- United States (active 2026 expansion market)
- UAE (active 2026 expansion market)

## Target job titles / personas
- D2C / e-commerce: Head of E-commerce, Director of Digital, VP Marketing, Head of Growth, Founder/CEO (at smaller D2C brands)
- Retail enterprise: Head of Omnichannel, Director of Digital Experience, Head of Customer Experience, CMO
- Real estate: Head of Marketing, Sales Director (developers); Head of Digital (brokerages)
- Automotive: Digital Marketing Lead, Head of Retail Experience, CX Lead
- Economic buyer: typically CMO or Head of E-commerce
- Champion: typically Digital / E-comm Manager feeling the pain of shoots and returns

## Pain points we solve
- Expensive, slow product photography and 3D production (one founder example: ₹1,10,000 spent on a shoot, 12 photos, only 3 units sold)
- High return rates from buyers who can't really see the product before buying (Ctruh has reported customer return rates dropping ~28% within weeks)
- Low PDP engagement and conversion from flat 2D images
- Technical complexity / need for engineering and 3D talent to build AR or XR (Ctruh is no-code, browser-native, no app downloads)
- Tool sprawl — brands stitching together separate vendors for 3D modeling, AR try-on, virtual stores, and ad creatives

## Disqualifiers (auto-low score)
- Brands with <10 SKUs or no online sales channel
- Pure services businesses with no physical product to visualize (unless real estate / education / healthcare use case)
- Companies in regions outside India, US, or UAE during current GTM focus
- B2B SaaS, fintech, or other categories with no visual / spatial product
- Brands with extremely low AOV where ROI on per-product 3D doesn't make sense
- Companies that already have a mature in-house 3D pipeline and dedicated XR team (lower urgency, harder displacement)
"""


def load_icp() -> str:
    """Load the ICP text, returning the default template if none saved yet."""
    if not ICP_PATH.exists():
        return DEFAULT_ICP_TEMPLATE
    try:
        data = json.loads(ICP_PATH.read_text(encoding="utf-8"))
        return data.get("icp", DEFAULT_ICP_TEMPLATE)
    except (json.JSONDecodeError, OSError):
        return DEFAULT_ICP_TEMPLATE


def save_icp(icp_text: str) -> None:
    """Persist the ICP text to disk."""
    ICP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ICP_PATH.write_text(
        json.dumps({"icp": icp_text}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
