"""
prompts.py
==========
All LangChain PromptTemplate definitions used in the pipeline.
No side-effects — pure data.
"""

from langchain_core.prompts import PromptTemplate

# ── Chitchat gate ─────────────────────────────────────────────
CHITCHAT_GATE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a classifier for a retail analytics assistant.
Decide whether the user input is:
  - "analytics" : any question about data, sales, orders, products, revenue, trends, categories, customers, reports, charts, statistics, or any request to show/display/analyze retail data.
               Also classify as "analytics" if it is a SHORT FOLLOW-UP or FILTER that clearly refines a previous analytics query (e.g. "for 2024 only", "خلال 2024 فقط", "والان لـ 2024", "top 5 only", "now filter by category").
  - "chitchat"  : greetings, general knowledge, history, geography, personal opinions, jokes, anything unrelated to the retail database.

Recent conversation history (use this to understand follow-up fragments):
{history}

User input: {question}

Respond with ONLY one word — either:  analytics   OR   chitchat
""",
)

# ── Chitchat response ─────────────────────────────────────────
CHITCHAT_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a friendly retail analytics assistant. The user has sent a non-analytics message.
Respond naturally and helpfully in the same language the user used.
If relevant, gently remind them that you can also answer retail data questions like sales trends, top products, revenue by category, etc.

User: {question}
Assistant:""",
)

# ── Query decomposition ───────────────────────────────────────
DECOMPOSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a retail data analyst. Decide if the following question is SIMPLE or COMPOUND.

SIMPLE  = a single aggregation, ranking, or lookup that can be answered in one pandas operation.
COMPOUND = requires 2+ separate data operations, such as:
  - Comparing two time periods (2024 vs 2025)
  - Finding top-N AND showing their trend over time
  - Growth rate / percentage change between periods
  - "Items that declined / grew" (requires comparison)
  - Any question with AND / ثم / مقارنة / مقابل / نسبة النمو

Respond ONLY with valid JSON — no markdown, no explanation.

Question: {question}

Return exactly this structure:
{{
  "is_compound": <true or false>,
  "steps": [
    "<step 1: precise English description of first sub-query>",
    "<step 2: precise English description of second sub-query>"
  ],
  "combination": "<merge_on_key | subtract | pct_change | display_separately | filter_by_step1>"
}}
""",
)

# ── Combine sub-step results ──────────────────────────────────
COMBINE_PROMPT = PromptTemplate(
    input_variables=["question", "steps", "combination", "step_results_info"],
    template="""
You are a senior data analyst combining multiple intermediate DataFrames into a final answer.

Original question: {question}
Sub-steps performed: {steps}
Combination method: {combination}
Available DataFrames and their columns:
{step_results_info}

Write ONLY raw executable Python/pandas code that:
1. Combines step_result_0, step_result_1, ... using the combination method.
2. Stores the final combined answer as a DataFrame called `result`.
3. Prints `result`.

- merge_on_key   → inner merge on shared key, keep all value columns
- subtract       → if inputs are scalars, construct a 2-row DataFrame with a 'Period' column and a 'Value' column so it can be plotted as a bar chart. If they are tables, merge on key and compute (value_step0 - value_step1)
- pct_change     → if inputs are scalars, construct a 2-row DataFrame with a 'Period' column and a 'Value' column, plus a 'pct_change' column. If tables, merge on key and compute ((value_step1 - value_step0) / value_step0 * 100).round(2), rename to 'growth_pct'
- display_separately → result = pd.concat([step_result_0, step_result_1], axis=0, ignore_index=True) if they share columns, else return step_result_0
- filter_by_step1 → use step_result_0 values as a filter on step_result_1

CRITICAL: NEVER return a DataFrame with just a single scalar difference (e.g. [-150000]). ALWAYS return the original values from BOTH steps as separate rows (e.g. row 1 = 2024, row 2 = 2025) so the result can be visually plotted!

No markdown, no fences, no comments.
""",
)

# ── Combined intent + decomposition (single LLM call) ─────────
INTENT_DECOMPOSE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a retail data analyst. Analyze the question and respond ONLY with valid JSON — no explanation, no markdown.

Question: {question}

PART 1 — INTENT (same rules as intent classification):
- ranking      : metric BROKEN DOWN by a categorical dimension (per category / per product / per brand / per seller /
                 لكل فئة / لكل منتج / حسب). Includes top-N / bottom-N / "highest" / "lowest" / "أكثر" / "أعلى" / "أدنى".
                 Use ranking whenever the answer is a sortable list of groups with one numeric measure.
                 Default chart_type for ranking is "hbar".
- trend        : metric evolution over time (month / year / quarter / week).
- comparison   : two specific periods or segments (2024 vs 2025, A vs B).
- distribution : share / proportion / percentage of a whole. chart_type "pie" or "donut".
- correlation  : relationship between two numeric variables.
- detail       : single-value lookup, aggregate KPI, or small specific record set. ONLY when none of the above fit.

PART 2 — DECOMPOSITION:
- SIMPLE  = single aggregation, ranking, or lookup answerable in one operation.
- COMPOUND = requires 2+ separate data operations:
    * Comparing two time periods (2024 vs 2025)
    * Top-N AND their trend over time
    * Growth rate / percentage change between periods
    * "Items that declined / grew" (requires comparison)
    * Any question with AND / ثم / مقارنة / مقابل / نسبة النمو
- ONLY EVER 2 STEPS. The combine stage supports exactly two sub-steps — NEVER emit 3+.
  A query over THREE OR MORE periods/months/segments per group (e.g. "compare Jan, Feb, Mar
  per category", "قارن يناير وفبراير ومارس لكل فئة") is SIMPLE: one query with the period
  column IN the GROUP BY (… WHERE d.month IN (1,2,3) GROUP BY category_name, d.month …) so the
  chart renders as a multi-line trend. Do NOT split it into per-month steps.

COMBINATION VALUES — choose strictly from this set (do NOT invent new values like "compare"):
- merge_on_key       : join two side-by-side periods/segments on a shared key (e.g. category) →
                       PREFERRED when intent is "comparison". The chart stage will render as a
                       grouped bar chart with both series in ONE plot.
- pct_change         : compute % change between two periods (growth rate questions).
- subtract           : compute difference between two periods.
- ratio              : compute step1 as a PERCENTAGE of step2 ((step1 / step2) * 100). Use for
                       "what % of the total is X", "نسبة X من الإجمالي", share-of-total questions.
                       Step 1 = the PART (e.g. one category), Step 2 = the WHOLE (e.g. all categories).
- filter_by_step1    : step 1 = the entity ranking (e.g. top-5 products/categories by revenue);
                       step 2 = the detail/trend over ALL entities (e.g. monthly revenue PER
                       product for the whole period). The COMBINE STAGE intersects them — step 2
                       must NOT mention step 1, "top 5", or any filter. The two steps run in
                       PARALLEL, so step 2 cannot see step 1. Write step 2 as a plain grouped
                       aggregation over every entity. Common when ranking + trend are combined.
                       CORRECT  step 1: "Total revenue per category in 2024"
                                step 2: "Monthly revenue per category in 2024"   ← no 'top 5', no filter
                       WRONG    step 2: "Monthly revenue for the top 5 categories from step 1"
                       ⚠️ ANY "trend / اتجاه / تطور / over time / monthly / شهري" in the question
                       MUST use filter_by_step1 with step 2 = a MONTHLY breakdown (include d.month
                       + d.month_name in step 2's grouping). NEVER use merge_on_key for a trend —
                       merge_on_key has no time axis and produces revenue_kwd_x/_y columns.
                       Both steps MUST name the entity column IDENTICALLY (e.g. both select
                       `p.name AS product_name` — never `name` in one step and `product_name` in
                       the other) so the combine can align them.
- display_separately : show two independent tables side-by-side. Use only when the two sub-queries
                       are answering genuinely different questions, not comparing the same metric.

CRITICAL — STEP SYMMETRY for merge_on_key / pct_change / subtract:
The two sub-steps MUST be SYMMETRIC: same metric, same grouping, only the time/segment differs.
The combine stage handles the comparison — never bundle the comparison into one step.

CORRECT (pct_change "Top 5 products by revenue growth 2024 → 2025"):
  Step 1: "Calculate total revenue per product in 2024"
  Step 2: "Calculate total revenue per product in 2025"
  combination: pct_change

WRONG (do NOT do this — breaks the combine stage):
  Step 1: "Calculate total revenue per product in 2024"
  Step 2: "Calculate revenue per product in 2025 AND the percent change"  ← combine is in step 2
  combination: pct_change

Same rule for merge_on_key and subtract. Each step is a single-period aggregation.
filter_by_step1 is the only asymmetric combination — step 1 produces a key set, step 2 uses it.

METRIC — what is being measured/aggregated. Required for cache disambiguation
("total revenue in 2024" and "total taxes in 2024" must NOT share a cache entry):
- revenue    : SUM(total_amount). Keywords: revenue, sales, إيرادات, مبيعات
- taxes      : SUM(tax_amount). Keywords: tax, taxes, ضريبة, ضرائب
- discount   : SUM(discount_amount). Keywords: discount, خصم, خصومات
- quantity   : SUM(quantity). Keywords: quantity, units, كمية, الكميات
- orders     : COUNT(DISTINCT order_id). Keywords: orders, طلب, طلبات
- items      : COUNT(*). Keywords: items, لين-ايتم, line items, rows
- unit_price : AVG(unit_price). Keywords: average price, متوسط السعر
- tax_rate   : AVG(tax_rate). Keywords: tax rate, نسبة الضريبة
- other      : when none of the above clearly applies

Return exactly this structure:
{{
  "intent_type":  "<ranking | trend | distribution | comparison | correlation | detail>",
  "chart_type":   "<hbar | line | vbar | pie | donut | histogram | scatter | area | table>",
  "needs_chart":  <true or false>,
  "top_n":        <integer or null>,
  "time_filter":  "<e.g. 2024, Q1 2024, or null>",
  "dimension":    "<product | category | subcategory | seller | customer | city | brand | total>",
  "metric":       "<revenue | taxes | discount | quantity | orders | items | unit_price | tax_rate | other>",
  "is_compound":  <true or false>,
  "steps":        ["<step 1: precise English description>", "<step 2: precise English description>"],
  "combination":  "<merge_on_key | subtract | pct_change | ratio | display_separately | filter_by_step1>"
}}
""",
)


# ── Query rewriter ────────────────────────────────────────────
REWRITE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a retail data analyst assistant. Rewrite the following user question into a clear, precise analytical intent in English.
Preserve all specific numbers, time filters, ranking limits, and entity names exactly as mentioned.
CRITICAL — entity names (brands, products, categories): keep them EXACTLY as the user wrote them,
in the ORIGINAL SCRIPT, and keep EVERY form the user gave. NEVER transliterate or translate an
Arabic brand/product name into Latin (keep تايد, اريال, بامبرز verbatim — do NOT produce "Taide",
"Taite", "Warial", "TIDE"). If the user wrote the brand in BOTH scripts like "داوني (DOWNY)",
KEEP BOTH terms in the rewrite — do not drop the Arabic one. The DB stores brands by Arabic name,
so a dropped Arabic term or a Latin guess will match nothing.
If the question is a SHORT FOLLOW-UP or FILTER (e.g. "for 2024 only", "خلال 2024 فقط", "والان لـ 2024"), use the conversation history to reconstruct the FULL analytical intent.
Do NOT answer the question — just rewrite it as a clear one-sentence English data analysis request.

Recent conversation history:
{history}

User question: {question}

Rewritten intent (one sentence):
""",
)

# ══════════════════════════════════════════════════════════════
# SQL PIPELINE (dwh1 star schema — new primary path)
# ══════════════════════════════════════════════════════════════

# ── SQL code generator ────────────────────────────────────────
SQL_PROMPT = PromptTemplate(
    input_variables=["schema_context", "question", "intent_hint", "history_context"],
    template="""
You are a senior data analyst writing PostgreSQL against a retail star-schema data warehouse.
Use ONLY the tables in the `dwh1` schema shown below.

SCHEMA CONTEXT (retrieved for this question):
{schema_context}

════════════════════════════════════════════════════
STAR SCHEMA CHEAT SHEET — dwh1
════════════════════════════════════════════════════

FACTS
  dwh1.fact_order_item                       -- main fact (233k rows); measures already pre-computed
    keys:    order_item_key (PK), customer_key, product_key, seller_key, category_key,
             brand_key, order_date_key, delivery_date_key, data_owner_key, order_id
    measures: unit_price, quantity, discount_amount, tax_amount, total_amount
    attrs:    order_status
    ⚠️ DATA NOTE: discount_amount is 0 for EVERY row in this dataset (no discounts recorded).
       Still write the correct SUM(f.discount_amount) query if asked about discounts — the result
       will simply be 0; do NOT add extra joins/filters trying to make it non-zero.

DIMENSIONS
  dwh1.dim_product     product_key, product_id, name (ar), en_name, price, tax_rate, sku, currency
  dwh1.dim_category    category_key, category_id, category_name (ar), sub_category_id, sub_category_name
                       -- BOTH levels live in ONE row. There is NO separate sub_categories table.
  dwh1.dim_brand       brand_key, brand_id, brand_name (ar), brand_en_name
  dwh1.dim_customer    customer_key, customer_id, name, email, phone, city   (PII currently all NULL)
  dwh1.dim_seller      seller_key,   seller_id,   seller_name, email, phone, city (PII currently all NULL)
  dwh1.dim_date        date_key (int yyyymmdd), full_date, day, day_name, month, month_name, year
  dwh1.dim_data_owner  data_owner_key, data_owner_id, data_owner_name (multi-tenant; usually 1 tenant)

⚠️ CRITICAL RULES
1. REVENUE: use SUM(f.total_amount) from fact_order_item. The measure is already
   unit_price*quantity net of discount + tax. Never compute SUM(unit_price*quantity) manually.
2. JOINS: always join facts to dims on the *_key surrogate columns:
      f.product_key   = p.product_key
      f.category_key  = c.category_key
      f.brand_key     = b.brand_key
      f.customer_key  = cu.customer_key
      f.seller_key    = s.seller_key
      f.order_date_key = d.date_key     (use d.year, d.month, d.month_name, d.full_date)
3. CATEGORIES: one join to dim_category gives you BOTH category_name and sub_category_name.
   Do NOT try to join a sub_categories table.
   ⚠️ GROUP AT THE LEVEL THE QUESTION ASKS — never mix the two levels:
   - "category / categories / فئة / فئات" (top categories, revenue by category, etc.)
     → SELECT and GROUP BY c.category_name ONLY. Do NOT add c.sub_category_name.
   - "sub-category / subcategory / فئة فرعية" → SELECT and GROUP BY c.sub_category_name.
   Adding the finer level to a category-level ranking silently changes the rows from
   N categories to N (category, sub-category) pairs — which is wrong. Pick exactly one.
4. DATES: filter on dim_date columns after joining (WHERE d.year = 2024, d.month = 6 etc).
   Do NOT parse date_key as a string.
   ALIAS EVERY AGGREGATE with a descriptive snake_case name — never leave a bare COUNT/SUM/AVG.
   e.g. COUNT(*) AS order_count, COUNT(DISTINCT order_id) AS order_count,
   SUM(f.quantity) AS quantity, AVG(f.unit_price) AS avg_unit_price. (A bare COUNT(*) yields a
   column literally named "count", which makes ugly chart labels.)
5. DELIVERY DATES: delivery_date_key = 10000000 is a sentinel for 'unknown'.
   Exclude it: WHERE f.delivery_date_key <> 10000000 before joining delivery dates.
6. ORDER STATUS lifecycle (pick based on the question):
      waiting              -- pending (bulk of rows: ~124k)
      invoiced             -- finalized (~101k)
      preparing / storekeeper_received / storekeeper_finished / delivered / done
   For "completed / actual sales" default to: order_status IN ('done','delivered','invoiced').
   If the question does not specify, DO NOT filter on order_status unless the user asked for
   completed/pending only.
7. CURRENCY: all monetary amounts are in Kuwaiti Dinar (KWD / دينار كويتي).
   - Alias revenue columns with _kwd suffix (e.g. revenue_kwd, spend_kwd).
   - Round to 2 decimal places: ROUND(SUM(f.total_amount)::numeric, 2) AS revenue_kwd.
   - NEVER use $, USD, JD, SAR.
8. LIMIT: always add LIMIT to prevent huge result sets. Default LIMIT 100. For ranking with
   an intent_hint top_n use LIMIT = top_n. Never exceed LIMIT 10000.
   ⚠️ EXCEPTION — "top N per <group>" / "top N in each <group>" / "أكثر N في كل <فئة>":
   DO NOT use a global LIMIT. Use ROW_NUMBER() over a window partitioned by the group:
       WITH ranked AS (
         SELECT p.name, c.category_name, SUM(f.quantity) AS qty,
                ROW_NUMBER() OVER (
                    PARTITION BY c.category_name
                    ORDER BY SUM(f.quantity) DESC
                ) AS rn
         FROM dwh1.fact_order_item f
         JOIN dwh1.dim_product p   ON f.product_key  = p.product_key
         JOIN dwh1.dim_category c  ON f.category_key = c.category_key
         GROUP BY p.product_key, p.name, c.category_name
       )
       SELECT name, category_name, qty
       FROM ranked
       WHERE rn <= 3
       ORDER BY category_name, qty DESC
       LIMIT 100;
   The window function gives N rows PER group instead of N rows total.
9. SAFETY: produce exactly ONE read-only SELECT statement. No INSERT/UPDATE/DELETE/DDL,
   no multiple statements, no semicolons inside string literals.
10. Qualify every column with its table alias (f., p., c., d., b., cu., s.).
    Use short aliases: fact_order_item=f, dim_product=p, dim_category=c, dim_brand=b,
    dim_customer=cu, dim_seller=s, dim_date=d, dim_data_owner=o.
11. GROUP BY: include every non-aggregated selected column.
12. For trend intent, ORDER BY the time columns you SELECTed/GROUPed BY, ascending
    (e.g. if you grouped by d.month only, ORDER BY d.month — NOT d.year).
    ⚠️ Every column in ORDER BY (and SELECT) of a grouped query must appear in GROUP BY
    or be inside an aggregate. Never ORDER BY a column that is only in the WHERE clause
    (a single-year filter means d.year is constant — do not order by it).
    For ranking intent, ORDER BY the aggregated measure DESC.
13. TEXT NAME FILTERS (category_name, sub_category_name, brand_name, product name, etc.):
    NEVER use exact `=` on a name — Arabic values often carry the definite article "ال"
    and spacing/spelling varies, so `= 'منظفات'` misses 'المنظفات' and returns 0 rows.
    ALWAYS match with case-insensitive ILIKE on a substring, stripping the leading article:
        WHERE c.category_name ILIKE '%منظفات%'
    Use the core noun WITHOUT "ال" inside the % … % so both 'منظفات' and 'المنظفات' match.
14. BRAND FILTERS — match against BOTH name columns so whichever script the user typed hits.
    `brand_name` is the Arabic name (COMPLETE for all brands: تايد, اريال, داوني, بامبرز, …);
    `brand_en_name` is the Latin name (HALF-EMPTY & partly misspelled — a useful bonus, never the
    sole filter). Use ILIKE on whatever term(s) the user gave, against both columns:
          WHERE (b.brand_name ILIKE '%داوني%' OR b.brand_en_name ILIKE '%DOWNY%')
    - If the user gave only one script, ILIKE that term against BOTH columns anyway
      (e.g. user wrote DOWNY → `b.brand_name ILIKE '%DOWNY%' OR b.brand_en_name ILIKE '%DOWNY%'`).
    - When the user wrote a brand in Arabic AND Latin like "داوني (DOWNY)", use BOTH terms —
      Arabic term on brand_name, Latin term on brand_en_name.
    - NEVER transliterate/guess a spelling that the user did not provide.

QUERY INTENT HINT:
{intent_hint}

CONVERSATION HISTORY:
{history_context}
════════════════════════════════════════════════════

User question: {question}

Write ONLY raw executable PostgreSQL. No markdown fences, no comments, no prose.
Return a single SELECT statement terminated by one semicolon.
""",
)

# ── SQL code fixer ────────────────────────────────────────────
SQL_FIX_PROMPT = PromptTemplate(
    input_variables=["sql", "error", "question", "schema_context"],
    template="""
You are a senior data analyst debugging PostgreSQL for a retail star-schema DWH (dwh1).
The following SQL was generated to answer: {question}

Relevant schema:
{schema_context}

CRITICAL REMINDERS:
- Revenue measure is already pre-computed: SUM(f.total_amount) from dwh1.fact_order_item.
- Join facts → dims on the surrogate *_key columns only.
- dim_category contains BOTH category and sub_category in one row. No sub_categories table.
- dim_date: date_key is YYYYMMDD integer. Filter on d.year/d.month after the join.
- delivery_date_key = 10000000 is a sentinel — exclude before joining delivery dates.
- Currency: KWD (Kuwaiti Dinar). Alias revenue columns with _kwd.
- Always qualify columns with the table alias (f., p., c., d., b., cu., s., o.).
- Exactly ONE read-only SELECT, with a LIMIT.
- EMPTY RESULT on a name filter usually means an exact `=` missed an Arabic article/spelling
  variant. Switch to ILIKE substring without the "ال" article: c.category_name ILIKE '%منظفات%'.
- GROUPING ERROR ("must appear in the GROUP BY clause"): every non-aggregated SELECT/ORDER BY
  column must be in GROUP BY. Do NOT ORDER BY a WHERE-only column (e.g. d.year under a year filter).

SQL that failed:
{sql}

Error: {error}

Return ONLY the corrected SQL — one SELECT, terminated with a semicolon. No markdown, no comments.
""",
)

# ── Combine fix prompt ────────────────────────────────────────
COMBINE_FIX_PROMPT = PromptTemplate(
    input_variables=["code", "error", "question", "step_results_info"],
    template="""
Fix this pandas combine code for: {question}

Available DataFrames:
{step_results_info}

Code that failed:
{code}

Error: {error}

Rewrite the code so it works. Store the final combined DataFrame as `result`.
No markdown, no fences, no comments.
""",
)

# ── Plotly chart generator ────────────────────────────────────
PLOTLY_PROMPT = PromptTemplate(
    input_variables=["question", "data_preview", "columns", "chart_hint"],
    template="""
You are a senior data visualization expert specializing in Arabic retail analytics.

User question: {question}
Result columns: {columns}
Data preview:
{data_preview}

CHART HINT: {chart_hint}

Chart rules:
- LINE  (px.line): trend over time — sort by time column first. For a trend BROKEN DOWN by a
  group (e.g. revenue per product/category over months) use ONE line per group via
  color='<group_col>'. Plot all groups in a single static chart.
- HBAR  (px.bar, orientation='h'): ranking / top-N — sort descending. Use x=numeric_col, y=name_col.
- VBAR  (px.bar): discrete group comparison with ≤ 6 categories.
- PIE   (px.pie): proportions ≤ 5 slices.
- DONUT (px.pie, hole=0.45): proportions > 5 slices.
- HISTOGRAM (px.histogram): continuous numeric distribution.
- SCATTER (px.scatter): correlation.
- NEVER use animation_frame / animated charts — a trend over time is always a STATIC line chart
  (use color=<group> for multiple series), never an animated/sliding bar.

⚠️ STRICT CHART HINT ENFORCEMENT:
- If CHART HINT says HBAR → you MUST write: px.bar(df, x='<numeric_col>', y='<name_col>', orientation='h')
  Sort the dataframe by the numeric column descending BEFORE plotting. Never use vbar for ranking.
- If chart has > 6 categories on x-axis → always use HBAR (horizontal), never VBAR.
- Each row must map to exactly ONE bar. Never stack or group unless explicitly asked.

MANDATORY:
- Produce EXACTLY ONE figure, assigned to a variable named `fig`. Do NOT create a second
  figure (no `fig_bar`, `fig_trend`, subplots) — one chart only, the single best one for the data.
- Apply fix_arabic() ONLY to Arabic STRING LITERALS you write (titles/labels). NEVER call
  fix_arabic() on the figure or on a dataframe column — `fix_arabic(fig)` is a fatal error.
- fig.update_layout(title_x=0.5)
- `fig` must stay a Plotly Figure for the whole script — never reassign it to a string/html
  (no `fig = fig.to_html(...)`, `fig = title`, `fig = fix_arabic(...)`). End with fig.show().
- Output RAW PYTHON ONLY — no prose, no explanations, no markdown fences before or after.
- CURRENCY: label all monetary axes/titles with "دينار كويتي (KWD)" — NEVER use $, USD, JD, or SAR.

COLORS — DO NOT OVERRIDE:
- A brand Plotly template ("namaa") is set as the process default. It supplies the colorway, fonts,
  background, gridlines, hover style, and color scales matching the website's Deep Indigo palette.
- DO NOT pass color, color_discrete_sequence, color_continuous_scale, marker_color, line_color,
  paper_bgcolor, plot_bgcolor, or font family to any plotly call. The template handles all of these.
- Only exception: if the data has a clearly positive vs negative meaning (e.g. growth_pct column),
  you may color bars conditionally with #059669 (positive) and #d4183d (negative).

fix_arabic() and `result` DataFrame are already in memory.
Write ONLY raw Python/plotly code. No markdown, no fences.
""",
)

# ── Plotly chart fixer ────────────────────────────────────────
PLOTLY_FIX_PROMPT = PromptTemplate(
    input_variables=["code", "error", "question", "data_preview", "columns"],
    template="""
Fix this plotly code for: {question}
Columns: {columns}
Data preview: {data_preview}
Error: {error}

Code: {code}

Common cause of "'str' object has no attribute ...": the code reassigned `fig` to a
string somewhere (e.g. `fig = fig.to_html(...)`, `fig = title`, or `fig = fix_arabic(...)`).
`fig` MUST stay a Plotly Figure for the whole script so `.update_layout()` / `.update_xaxes()`
work. Keep titles/labels in their own variables — never overwrite `fig`.

fix_arabic() and `result` DataFrame are available. Store the figure as `fig` and end with
fig.show(). Write ONLY raw Python/plotly code, no markdown.
""",
)

# ── Business recommendations ──────────────────────────────────
BUSINESS_RECO_PROMPT = PromptTemplate(
    input_variables=[
        "question",
        "data_preview",
        "columns",
        "intent_type",
        "accumulated_recommendations",
    ],
    template="""
You are a senior retail business analyst providing actionable recommendations to management.

Question answered: {question}
Intent: {intent_type}
Columns: {columns}
Data:
{data_preview}

ACCUMULATED RECOMMENDATIONS THIS SESSION:
{accumulated_recommendations}

Provide 3-5 concise, specific, actionable business recommendations.
- Reference specific numbers/names from the data.
- Build on accumulated recommendations — do NOT repeat same points.
- Be practical: stock, promotions, pricing, investigate, discontinue.
- CRITICAL LANGUAGE RULE: Detect the language from "Question answered" above ONLY. If it contains Arabic characters → write ALL recommendations in Arabic. If it is English → write entirely in English. NEVER mix languages. The language of ACCUMULATED RECOMMENDATIONS is irrelevant — ignore it when choosing your output language.
- Format as a numbered list.
- CURRENCY: always state monetary figures in Kuwaiti Dinar (KWD / دينار كويتي). Never use $, USD, JD, or SAR.
- NUMBER FORMATTING: round and abbreviate large numbers — English K/M ("1.37M", "127K"),
  Arabic ألف/مليون ("1.37 مليون", "127 ألف"). At most 1–2 significant decimals. Never write
  raw long figures like 1374522.71.
""",
)

# ── Follow-up question generator ──────────────────────────────
FOLLOWUP_PROMPT = PromptTemplate(
    input_variables=["question", "result_preview", "columns"],
    template="""
You are a retail analytics assistant. The user just asked: "{question}"

The result had these columns: {columns}
Result preview:
{result_preview}

Suggest exactly 3 follow-up questions that a retail manager would naturally ask next.
CRITICAL LANGUAGE RULE: If the original question contains Arabic characters → write ALL 3 questions in Arabic. If it is English → write all 3 in English. NEVER mix languages.
Each question must be specific (reference actual values or entities from the result).

Respond ONLY with a JSON array of 3 strings. No markdown, no explanation. Example:
["question 1", "question 2", "question 3"]
""",
)

# ── Reference resolver ────────────────────────────────────────
REFERENCE_RESOLVE_PROMPT = PromptTemplate(
    input_variables=["question", "history"],
    template="""
You are a retail analytics assistant. The user sent a follow-up query that may use pronouns,
implicit references, or be a SHORT FILTER/MODIFIER of a previous query.

Conversation history:
{history}

Follow-up question: {question}

Rewrite as a FULLY SELF-CONTAINED question if any of these apply:
- Contains vague references: "نفس", "same", "them", "it", "ذات", "هم", "هو", "هي", "ذلك", "تلك"
- Is a SHORT FRAGMENT that adds a filter/modifier to the previous query (e.g. "والان خلال 2024 فقط" → expand using the previous question's subject)
- Starts with "و" (Arabic "and") and references a previous query implicitly

If the question is already fully self-contained (has a clear subject and analytical intent), return it UNCHANGED.
Return ONLY the rewritten question — one sentence, no explanation.
""",
)

# ── Natural language summary ──────────────────────────────────
NL_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["question", "data_preview", "columns"],
    template="""
You are a retail analytics assistant. Summarize the data result below in exactly 2-3 clear, concise sentences.
Reference specific numbers, entities, and patterns visible in the data.
Be direct and insightful — no filler phrases like "the table shows" or "as we can see".
CRITICAL LANGUAGE RULE: Detect the language of the question below. If it contains Arabic characters → write the ENTIRE summary in Arabic. If it is English → write entirely in English. NEVER mix languages. NEVER translate the question.
CURRENCY: use Kuwaiti Dinar (KWD / دينار كويتي) for any monetary value.
ALL-ZERO DATA: if every value of the asked metric is 0 (e.g. discounts — none are recorded in this
dataset), say so plainly ("لا توجد خصومات مسجلة في البيانات" / "no discounts are recorded in the data")
instead of listing zeros. Do NOT claim the data is missing or that the query failed — the data simply has none.
NUMBER FORMATTING — make numbers easy to read, never write raw long figures:
- Round and abbreviate large numbers. English: 1,374,522.71 → "1.37M", 127,404 → "127K", 8,640 → "8.6K".
- Arabic: use مليون / ألف — e.g. 1374522.71 → "1.37 مليون", 127404 → "127 ألف", 8640 → "8.6 ألف".
- Keep at most 1–2 significant decimals (e.g. 1.37M, not 1.374522M). Whole-ish values drop decimals.
- Percentages: one decimal max (e.g. "‎-9.3%"). Small counts (< 1000) may stay as-is.

Question: {question}
Columns: {columns}
Data:
{data_preview}

Summary (2-3 sentences only):
""",
)

# ── Chart edit gate ───────────────────────────────────────────
CHART_EDIT_GATE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are a classifier. Decide if the user input is a request to MODIFY the currently displayed chart
(e.g. change chart type, flip orientation, change colors, sort differently, add/remove labels,
zoom in on a region, switch to log scale, highlight a specific bar or line)
OR if it is a new data query that requires fetching new data.

User input: {question}

Respond ONLY with one word:  chart_edit   OR   new_query
""",
)

# ── Chart edit code generator ─────────────────────────────────
CHART_EDIT_PROMPT = PromptTemplate(
    input_variables=["instruction", "plotly_code", "columns", "data_preview"],
    template="""
You are a senior data visualization expert. The user wants to modify an existing chart.

Instruction: {instruction}
DataFrame columns available: {columns}
Data preview:
{data_preview}

Existing plotly code to modify:
{plotly_code}

Apply the SMALLEST possible change to the existing code that satisfies the instruction —
treat it as a surgical diff, not a rewrite.
- Start from the existing code EXACTLY as given and change ONLY what the instruction asks.
- Preserve every other argument verbatim: the same x/y columns, the same `color=` argument
  (keep it if present, do not add or remove it), the same data prep / sort / groupby lines,
  the same titles, axis labels, ordering, height, and barmode.
- "flip" / "اقلب" / "اجعله أفقي/عمودي" means ONLY swap orientation (x↔y and
  orientation='h'↔'v'); keep the same columns, color, and sort. Do not re-pick columns.
- Do NOT drop the legend/color grouping, do NOT re-aggregate, do NOT re-sort unless the
  instruction explicitly says so.
- Apply fix_arabic() on any NEW Arabic text strings you add.
- Store the figure as `fig` (never reassign `fig` to a string). End with fig.show().
- fix_arabic() and the `result` DataFrame are already in memory.
Write ONLY raw Python/plotly code. No markdown, no fences, no comments.
""",
)

# ── Spell / typo correction ───────────────────────────────────
SPELL_CORRECT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
You are an Arabic/English text correction assistant.
Fix any obvious spelling or typographic mistakes in the following query.
Do NOT change numbers, names, entity names, or meaning — only fix clear misspellings.
If the text is already correct, return it UNCHANGED.
Respond with ONLY the corrected text — no explanation, no extra words.

Query: {question}

Corrected:""",
)

# ── Summary fallback ──────────────────────────────────────────
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["question", "error", "schema_hint"],
    template="""
You are a retail analytics expert. A technical query could not be executed automatically.

User question: {question}
Error: {error}
Schema hint: {schema_hint}

Provide a helpful TEXT-ONLY answer:
- Explain what the data analysis would typically show.
- Suggest how to rephrase the question for better results.
- If possible, give a rough qualitative answer based on general retail knowledge.

Respond in the SAME LANGUAGE as the question. Be concise (2-4 sentences).
""",
)
