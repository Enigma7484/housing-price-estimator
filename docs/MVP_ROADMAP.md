# MVP Roadmap and Sales Positioning

## Recommended Pivot

Reposition the project from **AI Estimator Platform** to **AI Valuation and Quote Copilot**.

The current app already has the bones of this product: multiple valuation models, typed API contracts, a model registry, and a dashboard. The commercial version should not feel like a collection of demos. It should help a buyer answer one painful business question:

> "What should we quote, offer, list, or approve, and why?"

## Best First Market

### Primary MVP: Resale and Trade-In Valuation Copilot

Target users:

- Used electronics refurbishers
- Phone repair shops
- Small device resellers
- Local buy/sell businesses
- Used car dealers as a later vertical

Why this market fits the current app:

- The app already has mobile, car, and housing estimators.
- Mobile/device resale is easier to MVP than real estate or insurance because the input schema is smaller and the sales cycle can be shorter.
- Resale businesses care about fast, consistent offers, margin protection, and explaining why a number was chosen.
- The existing mobile estimator can be converted from "price range classification" into a resale quote workflow.

## Market Signals

- CPQ and quote automation are growing because teams want faster, more accurate quotes, approval control, pricing rules, and automated proposal generation. Salesforce describes quoting software as a way to generate and manage quotes with product catalogs, discount controls, tax/currency handling, templates, approvals, and automated calculations: https://www.salesforce.com/ap/sales/revenue-lifecycle-management/automated-quoting-software/
- Persistence Market Research estimates the CPQ market at USD 3.2B in 2025 and USD 8.9B by 2032, a 15.7% CAGR: https://www.persistencemarketresearch.com/market-research/configure-price-quote-software-market.asp
- Mordor Intelligence reports that AI-driven dynamic pricing held 47.4% of the price optimization software market in 2025, with cloud deployment also dominant: https://www.mordorintelligence.com/industry-reports/price-optimization-software-market
- The used/refurbished smartphone market is projected by Mordor Intelligence to grow from USD 65.20B in 2025 to USD 96.99B by 2031: https://www.mordorintelligence.com/industry-reports/used-and-refurbished-smartphone-market
- The second-hand electronics market was estimated at USD 139.5B in 2025 by Global Market Insights, with projected growth through 2035: https://www.gminsights.com/industry-analysis/second-hand-electronic-products-market
- Automotive AI appraisal and inventory tools are being adopted because dealers need to avoid overpaying, price inventory correctly, and reduce manual repricing. CDK Global frames predictive AI as a way to identify vehicles to acquire and avoid overpaying: https://www.cdkglobal.com/insights/why-predictive-ai-crucial-used-car-inventory-2025

## Buyer Pain Points

### Resale and Refurbishment

- Staff quote inconsistently from memory or spreadsheets.
- A bad buy price destroys margin before the item is even listed.
- Market prices move faster than manual pricing sheets.
- Customers challenge trade-in offers and need a clear explanation.
- Small operators cannot afford enterprise pricing systems.
- Owners need a record of who quoted what and why.

### Used Auto

- Managers overpay on trades when relying on intuition.
- Aged inventory forces markdowns.
- Market volatility makes yesterday's price wrong today.
- Staff need price ranges, confidence bands, and comps-style justification.

### Home Services and Contractors

- Manual estimates are slow and error-prone.
- Missed labor/material costs erase profit.
- Customers expect fast, professional quotes.
- Owners need guardrails on discounting and change orders.

## Product Definition

### Name Options

- QuotePilot AI
- ResaleIQ
- ValuMate
- PriceDesk AI
- MarginGuard

Recommended MVP name: **ResaleIQ**

Tagline:

> AI trade-in and resale pricing for shops that need fast offers without guessing.

### Core Job

Help a shop turn product details into a defensible offer, list price, and expected margin.

### MVP Workflow

1. Staff selects item type: phone, laptop, tablet, car later.
2. Staff enters model/specs/condition.
3. App returns:
   - Recommended buy offer
   - Recommended resale/list price
   - Expected gross margin
   - Confidence level
   - Key pricing factors
   - Quote notes for the customer
4. Staff saves or exports the quote.
5. Owner sees quote history, staff activity, and margin risk.

## What To Build From Current

### Keep

- FastAPI backend
- React frontend
- Estimator registry
- Existing model artifacts as demo seed
- Dashboard concept
- Modular estimator architecture

### Change

- Rename the product from platform/demo language to a buyer-specific product.
- Replace "Housing/Mobile/Car Estimator" navigation with "Quote", "Inventory", "History", "Dashboard", and "Settings".
- Make mobile/device resale the primary app.
- Move housing into a demo/secondary module unless targeting real estate.
- Convert model outputs into business decisions: buy price, list price, margin, confidence, explanation.
- Add saved quotes and CSV export before adding more ML models.

## MVP Scope

### Week 1: Commercial Reframe

- Rename UI to ResaleIQ or QuotePilot AI.
- New first screen: quote workspace, not landing page.
- Replace estimator catalog with vertical-specific modules.
- Add customer-facing quote summary copy.
- Add mock quote history using local state or simple JSON-backed storage.

Acceptance criteria:

- A buyer can understand the app in 10 seconds.
- The demo starts with a real workflow, not a model catalog.

### Week 2: Device Resale Workflow

- Add fields for device model, storage, carrier lock, cosmetic condition, battery health, accessories, and repair status.
- Convert mobile result into buy offer, resale price, and margin.
- Add confidence band and factor explanations.
- Add "copy quote" and "download CSV" actions.

Acceptance criteria:

- A shop can generate a quote they could plausibly send to a customer.
- The app explains the price in plain language.

### Week 3: Owner Dashboard

- Add quote history table.
- Add average margin, quotes by status, high-risk quotes, and staff/user placeholder.
- Add manual override reason.
- Add settings for target margin, max offer percentage, and condition adjustment.

Acceptance criteria:

- The owner can see whether pricing behavior is protecting margin.
- Manual overrides are tracked.

### Week 4: Sellable Demo Package

- Add seeded demo data.
- Add a polished pitch route: `/pitch`.
- Add one-click demo scenarios: "iPhone trade-in", "gaming laptop", "used sedan".
- Add README sales positioning and demo script.
- Deploy frontend/backend.

Acceptance criteria:

- You can demo the full product in under five minutes.
- The demo tells a business story, not a technical story.

## Pricing Strategy

### Validation Pricing

- Solo shop: USD 49/month
- Small shop: USD 149/month for up to 5 users
- Multi-location: USD 399/month+

Start with founder-led pilots:

- USD 500 to USD 1,500 setup for a custom pricing sheet import
- First month free only if they agree to two feedback calls
- Discount annual plans only after product-market signal

### What Makes It Worth Paying For

- Saves staff time on each quote.
- Reduces underpriced resale listings.
- Prevents overpaying on trade-ins.
- Creates a repeatable pricing process.
- Gives owners visibility into pricing decisions.

## Pitch

### One-Liner

ResaleIQ helps resale shops generate consistent AI-backed trade-in offers and resale prices in seconds, with margin guardrails and plain-English explanations.

### 30-Second Pitch

Small resale shops still price trade-ins from memory, spreadsheets, and marketplace checks. That makes quotes slow, inconsistent, and risky for margin. ResaleIQ turns device details and condition into a recommended buy offer, resale price, confidence score, and customer-ready explanation. Owners get a dashboard of quote history, margin risk, and overrides, so pricing becomes a repeatable workflow instead of a guessing game.

### Demo Script

1. "A customer brings in an iPhone."
2. "The staff member enters model, storage, condition, battery health, and accessories."
3. "The app recommends a buy offer and resale price."
4. "It shows expected margin and the factors behind the recommendation."
5. "The staff member copies a customer-ready quote."
6. "The owner can review quote history and override patterns."

## Positioning Against Alternatives

### Not Another Generic Estimator

Generic estimators predict a number. ResaleIQ turns the number into an operating decision.

### Not Enterprise CPQ

Enterprise CPQ is broad and expensive. ResaleIQ is narrower: fast resale quotes, margin controls, and explainable offer logic for small operators.

### Not Just a Spreadsheet

Spreadsheets do not give confidence, explanations, quote history, override tracking, or a polished workflow.

## App Ideas Ranked

1. **ResaleIQ: Device Trade-In and Resale Pricing**
   - Best fit with current app.
   - Clear buyer.
   - Shortest path to MVP.
   - Good monthly SaaS potential.

2. **DealerDesk AI: Used Car Appraisal Copilot**
   - Bigger budgets.
   - More competitive and data-dependent.
   - Good second vertical after proving the quote workflow.

3. **QuotePilot AI: Contractor Estimate Assistant**
   - Strong pain point.
   - Requires materials/labor database, templates, and vertical focus.
   - Better as a separate product if you want home services.

4. **RentRight AI: Rental Pricing Assistant**
   - Useful, but data and compliance sensitivity are higher.
   - More difficult to sell without live market comps.

5. **HomeValue Copilot**
   - Existing housing model helps, but the market is crowded and buyers expect comps, maps, and local data.

## Final Recommendation

Build and pitch **ResaleIQ** first.

The current app should become a vertical SaaS demo for resale pricing. Keep the broader estimator architecture under the hood, but sell one specific workflow: fast, consistent trade-in quotes with margin protection.

The next code milestone should be a UI restructure around quote generation and history, using the current mobile estimator as the first production-style workflow.
