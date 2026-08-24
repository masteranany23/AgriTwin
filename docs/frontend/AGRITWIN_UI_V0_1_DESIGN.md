# AgriTwin UI v0.1 — Design Brief

## 1. Goal

Create a simple, professional and scientifically credible web interface
for demonstrating AgriTwin to government officials, researchers and
institutional stakeholders.

The UI should communicate three things clearly:

1. What is happening in a field now.
2. How AgriTwin learns from real observations.
3. How the updated digital twin forecasts the future.

The UI is an official-facing demonstrator, not the complete AgriTwin platform.

---

## 2. Core Message

### Headline

A Digital Twin for Every Field.

### Supporting message

AgriTwin combines physics-based crop simulation, real-world observations
and sequential data assimilation to understand field conditions and
forecast what happens next.

### Core visual story

Observe → Fuse → Simulate → Assimilate → Forecast

---

## 3. Visual Style

The design should feel:

- Professional
- Scientific
- Clean
- Modern
- Calm
- Agricultural
- Trustworthy
- Premium but restrained

Avoid:

- Generic SaaS dashboard appearance
- Excessive green
- Excessive gradients
- Heavy glassmorphism
- Neon colours
- Cartoon farming graphics
- Excessive animation
- Clutter

Use generous whitespace and clear visual hierarchy.

---

## 4. Colour Direction

Primary:
Deep forest green

Secondary:
Muted sage

Background:
Warm off-white

Surface:
White

Text:
Dark charcoal

Accent:
Subtle earth / amber

The interface should not be overwhelmingly green.

---

## 5. Typography

Use a modern sans-serif font such as:

- Inter
- Manrope

Use one primary font family if possible.

Typography should be clean, spacious and highly readable.

---

## 6. Page Structure

The initial website should be a polished single-page experience.

Sections:

1. Navigation
2. Hero
3. Field Digital Twin
4. Current vs Future State
5. Assimilation Timeline
6. Open Loop vs Assimilated Chart
7. How AgriTwin Works
8. Scientific Foundation
9. Footer

---

## 7. Hero

Eyebrow:

AGRICULTURAL DIGITAL TWIN

Headline:

A Digital Twin for Every Field.

Description:

AgriTwin combines physics-based crop simulation, real-world observations
and sequential data assimilation to understand field conditions and
forecast what happens next.

Primary CTA:

Explore the Digital Twin

Secondary CTA:

How It Works

The hero should contain an elegant visual representation of a crop field.

Do not use a generic stock photograph as the main visual.

---

## 8. Field Digital Twin

This is the visual centerpiece of the website.

Represent a crop field using a clean 2D/SVG-style visualization.

The field should contain:

- Crop rows
- Individual plants
- Field boundary
- Subtle soil representation
- Small observation markers

Plant density and size should visually suggest crop development / LAI.

The visualization should support:

- Current state
- Assimilated state
- Future / forecast state

A future version may use Three.js/WebGL, but v0.1 should prioritize
a polished 2D experience.

---

## 9. Demonstration Data

Use deterministic demonstration data initially.

Field:
Demonstration Field

Crop:
Wheat

Simulation Day:
42

Current:

LAI: 2.84
Soil Moisture: 31%
Biomass: 4.21 t/ha
DVS: 0.74

Observation:

Source: Sentinel-2
Observed LAI: 2.61
Confidence: 91%

Assimilated:

LAI: 2.67

Forecast:

LAI: 3.42
Projected Yield: 4.80 t/ha
Forecast Confidence: 87%

These values are illustrative demonstration data and must not be presented
as real field measurements.

---

## 10. Assimilation Demonstration

The main interactive story should be:

Current Field
↓
Observation Arrives
↓
Data Fusion
↓
EnKF Assimilation
↓
Updated Digital Twin
↓
Future Forecast

The field visualization, metrics and chart should update during this sequence.

The animation should be subtle and scientifically credible.

---

## 11. Charts

Include one clean time-series chart comparing:

- Open-loop simulation
- Assimilated simulation
- Observations

The chart should communicate that observations modify the trajectory of
the digital twin.

Avoid excessive chart decoration.

---

## 12. How AgriTwin Works

Show a simple visual pipeline:

Observe
↓
Quality Control
↓
Data Fusion
↓
Physics-Based Simulation
↓
EnKF
↓
Forecast

Technical details should remain secondary.

The first-time visitor should understand the concept without needing
knowledge of WOFOST or EnKF.

---

## 13. Animation

Use subtle animations only.

Preferred:

- Section fade/reveal
- Small card hover elevation
- Smooth number transitions
- Field state transition
- Assimilation sequence
- Chart transitions

Avoid:

- Bouncing
- Excessive parallax
- Spinning elements
- Constant floating objects
- Particle-heavy backgrounds
- Distracting effects

Animation should communicate state changes rather than exist only for decoration.

---

## 14. Responsive Design

Desktop:
Large field visualization and spacious layouts.

Tablet:
Reduced spacing and simplified layouts.

Mobile:
Single-column layout.

Mobile order:

Hero
↓
Field
↓
Current State
↓
Future State
↓
Assimilation
↓
Chart
↓
How It Works
↓
Science

No horizontal scrolling.

---

## 15. Technology Direction

Preferred eventual frontend stack:

React
TypeScript
Vite
Tailwind CSS
shadcn/ui
Motion
Recharts

The frontend should remain independent and exportable.

GitHub remains the source of truth.

Replit may be used as the initial development environment.

The frontend must not duplicate scientific calculations from the backend.

---

## 16. Out of Scope for v0.1

Do not build yet:

- Authentication
- User accounts
- Admin dashboard
- Complex GIS
- Multiple field management
- IoT management
- Scenario optimization
- Alerts
- Advanced reports
- Payment systems
- Chatbot

These will be added in future versions.

---

## 17. Design Success Criteria

The finished interface should allow a government official to understand
within approximately 30 seconds:

"What is happening in this field?"

"What did AgriTwin learn from observations?"

"How did the digital twin update?"

"What does AgriTwin predict next?"

The interface should feel like a serious scientific / climate-tech product,
not a generic agricultural dashboard.