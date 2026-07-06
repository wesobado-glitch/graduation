# NAMAA Analytics Agent — 200 Test Queries

> Covers: Arabic, English, KPI, ranking, trend, distribution, comparison, compound,
> chart-edit, semantic cache, exact cache, chitchat, follow-up references, edge cases.

---

## 1. Revenue & Financial KPIs (20 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 1  | ما إجمالي الإيرادات؟ | KPI AR | KPI card |
| 2  | What is the total revenue? | KPI EN | KPI card |
| 3  | كم إجمالي الإيرادات في 2024؟ | KPI AR | KPI card |
| 4  | Total revenue in 2025 | KPI EN | KPI card |
| 5  | ما متوسط قيمة الطلب؟ | KPI AR | KPI card |
| 6  | What is the average order value? | KPI EN | KPI card |
| 7  | إجمالي الضرائب المحصلة في 2024 | KPI AR | KPI card |
| 8  | Total discount amount given in 2025 | KPI EN | KPI card |
| 9  | كم عدد الطلبات الإجمالية في البيانات؟ | KPI AR | KPI card |
| 10 | How many unique orders exist in the data? | KPI EN | KPI card |
| 11 | ما إجمالي الكميات المباعة؟ | KPI AR | KPI card |
| 12 | Total quantity sold in 2024 | KPI EN | KPI card |
| 13 | كم إجمالي الإيرادات في الربع الأول من 2025؟ | KPI AR | KPI card |
| 14 | Revenue in Q4 2024 | KPI EN | KPI card |
| 15 | ما إجمالي الخصومات في 2024؟ | KPI AR | KPI card |
| 16 | What percentage of orders have a discount? | KPI EN | KPI card |
| 17 | كم إجمالي إيرادات شهر يناير 2025؟ | KPI AR | KPI card |
| 18 | Total revenue for March 2024 | KPI EN | KPI card |
| 19 | ما نسبة الضريبة المتوسطة على المنتجات؟ | KPI AR | KPI card |
| 20 | How many products have been sold at least once? | KPI EN | KPI card |

---

## 2. Product Analysis (20 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 21 | أكثر 10 منتجات تحقيقاً للإيرادات | Ranking AR | Table + HBar |
| 22 | Top 5 products by revenue in 2024 | Ranking EN | Table + HBar |
| 23 | أقل 5 منتجات مبيعاً من حيث الكمية | Ranking AR | Table + HBar |
| 24 | Bottom 10 products by quantity sold | Ranking EN | Table + HBar |
| 25 | ما المنتجات التي لم تُباع في 2024؟ | Detail AR | Table |
| 26 | Which products were sold only once? | Detail EN | Table |
| 27 | أكثر 5 منتجات من حيث عدد الطلبات في 2025 | Ranking AR | Table + HBar |
| 28 | Top 10 products by number of orders in 2025 | Ranking EN | Table + HBar |
| 29 | ما متوسط سعر المنتجات؟ | KPI AR | KPI card |
| 30 | Show me all products priced above 500 KWD | Detail EN | Table |
| 31 | أكثر المنتجات تحقيقاً للإيرادات في يناير 2025 | Ranking AR | Table + HBar |
| 32 | Top 5 products by discount amount given | Ranking EN | Table + HBar |
| 33 | المنتجات التي تجاوزت إيراداتها 100 ألف دينار | Detail AR | Table |
| 34 | Products with quantity sold greater than 1000 units | Detail EN | Table |
| 35 | أكثر 10 منتجات من حيث الكميات المباعة في 2024 | Ranking AR | Table + HBar |
| 36 | Show products with zero discount in 2025 | Detail EN | Table |
| 37 | ما المنتجات الأكثر ربحية بعد خصم الضريبة؟ | Ranking AR | Table + HBar |
| 38 | Top 5 products by revenue growth from 2024 to 2025 | Compound EN | Table + Chart |
| 39 | قارن مبيعات أكثر 5 منتجات في 2024 و 2025 | Compound AR | Table + Chart |
| 40 | أكثر 3 منتجات مبيعاً في كل فئة | Ranking AR | Table |

---

## 3. Category Analysis (20 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 41 | إجمالي الإيرادات لكل فئة رئيسية | Distribution AR | Table + HBar |
| 42 | Revenue by category in 2024 | Distribution EN | Table + HBar |
| 43 | أكثر 5 فئات إيرادات في 2025 | Ranking AR | Table + HBar |
| 44 | Top 3 categories by number of orders | Ranking EN | Table + HBar |
| 45 | ما الفئة الأكثر مبيعاً من حيث الكمية؟ | KPI AR | KPI card |
| 46 | Which category has the lowest revenue? | KPI EN | KPI card |
| 47 | إيرادات فئة المنظفات شهرياً في 2024 | Trend AR | Table + Line |
| 48 | Monthly revenue for المواد الغذائية in 2025 | Trend EN | Table + Line |
| 49 | قارن إيرادات فئة العناية الشخصية بين 2024 و 2025 | Comparison AR | Table + Chart |
| 50 | Compare المنظفات vs المناديل revenue in 2024 | Comparison EN | Table + Chart |
| 51 | نسبة مساهمة كل فئة في إجمالي الإيرادات | Distribution AR | Table + Pie |
| 52 | Category revenue share as percentage of total | Distribution EN | Table + Pie |
| 53 | أي فئة شهدت أكبر نمو من 2024 إلى 2025؟ | Ranking AR | Table + HBar |
| 54 | Category with highest average order value | KPI EN | KPI card |
| 55 | إيرادات فئة الكترونيات ربع سنوية في 2025 | Trend AR | Table + Line |
| 56 | Show orders count per category in 2025 | Distribution EN | Table + HBar |
| 57 | أكثر الفئات التي تحصل على خصومات | Ranking AR | Table + HBar |
| 58 | Which subcategory has the most revenue? | Ranking EN | Table + HBar |
| 59 | إجمالي الكميات المباعة لكل فئة في 2024 | Distribution AR | Table + HBar |
| 60 | Revenue per category for Q1 2025 | Distribution EN | Table + HBar |

---

## 4. Brand Analysis (20 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 61 | أكثر 10 علامات تجارية إيرادات | Ranking AR | Table + HBar |
| 62 | Top 5 brands by revenue in 2025 | Ranking EN | Table + HBar |
| 63 | إيرادات علامة تايد (TIDE) شهرياً في 2024 | Trend AR | Table + Line |
| 64 | Monthly revenue for PAMPERS in 2025 | Trend EN | Table + Line |
| 65 | قارن إيرادات تايد واريال في 2024 | Comparison AR | Table + Chart |
| 66 | Compare CLOROX vs DETTOL revenue in 2025 | Comparison EN | Table + Chart |
| 67 | أي علامة تجارية لديها أعلى متوسط سعر؟ | KPI AR | KPI card |
| 68 | Which brand has the highest number of orders? | KPI EN | KPI card |
| 69 | إيرادات العلامات التجارية في فئة المنظفات | Distribution AR | Table + HBar |
| 70 | Top 5 brands in العناية بالطفل category | Ranking EN | Table + HBar |
| 71 | نمو إيرادات داوني (DOWNY) من 2024 إلى 2025 | Comparison AR | Table |
| 72 | Brand revenue growth from 2024 to 2025 for top 5 | Compound EN | Table + Chart |
| 73 | أكثر العلامات التجارية التي تحصل على خصومات | Ranking AR | Table + HBar |
| 74 | Which brand sells the most units per order? | Ranking EN | Table |
| 75 | إيرادات فانيش (VANISH) ربع سنوية في 2025 | Trend AR | Table + Line |
| 76 | Top 10 brands by quantity sold in 2024 | Ranking EN | Table + HBar |
| 77 | ما العلامات التجارية الأقل مبيعاً في 2025؟ | Ranking AR | Table + HBar |
| 78 | Average order value per brand | Ranking EN | Table + HBar |
| 79 | أي علامة تجارية تحقق أعلى إيرادات في يناير 2025؟ | Ranking AR | Table |
| 80 | Brands with revenue above 1 million KWD | Detail EN | Table |

---

## 5. Order Status Analysis (15 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 81 | كم عدد الطلبات في كل حالة؟ | Distribution AR | Table + Pie |
| 82 | Show order count by status | Distribution EN | Table + Pie |
| 83 | ما إجمالي إيرادات الطلبات المكتملة (done)؟ | KPI AR | KPI card |
| 84 | Total revenue of invoiced orders | KPI EN | KPI card |
| 85 | ما نسبة الطلبات في حالة انتظار (waiting)؟ | KPI AR | KPI card |
| 86 | Which order status generates the most revenue? | Ranking EN | Table + HBar |
| 87 | إيرادات الطلبات المسلمة (delivered) في 2025 | KPI AR | KPI card |
| 88 | Monthly trend of waiting orders in 2025 | Trend EN | Table + Line |
| 89 | عدد الطلبات المكتملة شهرياً في 2024 | Trend AR | Table + Line |
| 90 | Compare done vs invoiced orders revenue in 2025 | Comparison EN | Table + Chart |
| 91 | ما الفئة التي لديها أعلى نسبة طلبات مكتملة؟ | Ranking AR | Table |
| 92 | Orders by status per category in 2025 | Distribution EN | Table |
| 93 | ما المنتجات التي لها أكثر طلبات في حالة انتظار؟ | Ranking AR | Table |
| 94 | Average order value for done vs waiting status | Comparison EN | Table + Chart |
| 95 | تطور حالات الطلبات شهرياً في 2025 | Trend AR | Table + Line |

---

## 6. Trend & Time Series (25 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 96  | تطور المبيعات الشهرية في 2024 | Trend AR | Table + Line |
| 97  | Monthly revenue trend in 2025 | Trend EN | Table + Line |
| 98  | تطور عدد الطلبات أسبوعياً في يناير 2025 | Trend AR | Table + Line |
| 99  | Daily revenue for February 2025 | Trend EN | Table + Line |
| 100 | المبيعات ربع السنوية في 2024 | Trend AR | Table + Line |
| 101 | Quarterly revenue comparison 2024 vs 2025 | Trend EN | Table + Line |
| 102 | تطور إيرادات فئة المنظفات شهرياً في 2025 | Trend AR | Table + Line |
| 103 | Monthly units sold trend in 2024 | Trend EN | Table + Line |
| 104 | تطور متوسط قيمة الطلب شهرياً في 2025 | Trend AR | Table + Line |
| 105 | Revenue trend for PAMPERS brand in 2025 | Trend EN | Table + Line |
| 106 | أي شهر حقق أعلى إيرادات في 2024؟ | KPI AR | KPI card |
| 107 | Which quarter had the highest orders in 2025? | KPI EN | KPI card |
| 108 | تطور الخصومات المقدمة شهرياً في 2024 | Trend AR | Table + Line |
| 109 | Weekly order count for March 2025 | Trend EN | Table + Line |
| 110 | مقارنة المبيعات اليومية بين يناير وفبراير 2025 | Comparison AR | Table + Chart |
| 111 | Revenue growth month over month in 2025 | Trend EN | Table + Line |
| 112 | تطور مبيعات أكثر 3 فئات شهرياً في 2025 | Compound AR | Table + Chart |
| 113 | Monthly revenue trend for top 3 brands in 2024 | Compound EN | Table + Chart |
| 114 | أي يوم من الأسبوع يحقق أعلى مبيعات؟ | Distribution AR | Table + HBar |
| 115 | Which day of the week has the most orders? | Distribution EN | Table + HBar |
| 116 | تطور الطلبات المكتملة شهرياً في 2025 | Trend AR | Table + Line |
| 117 | Compare H1 vs H2 revenue in 2024 | Comparison EN | Table + Chart |
| 118 | تطور الكميات المباعة ربع سنوياً في 2025 | Trend AR | Table + Line |
| 119 | Revenue trend for المواد الغذائية monthly 2025 | Trend EN | Table + Line |
| 120 | أي شهر شهد أعلى كمية مباعة في 2025؟ | KPI AR | KPI card |

---

## 7. Comparison Queries (20 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 121 | قارن إجمالي المبيعات بين 2024 و 2025 | Comparison AR | Table + Chart |
| 122 | Compare total revenue 2024 vs 2025 | Comparison EN | Table + Chart |
| 123 | قارن إيرادات الربع الأول من 2024 بالربع الأول من 2025 | Comparison AR | Table + Chart |
| 124 | Compare Q2 2024 vs Q2 2025 revenue | Comparison EN | Table + Chart |
| 125 | مقارنة عدد الطلبات في 2024 مقابل 2025 | Comparison AR | Table + Chart |
| 126 | Compare المنظفات vs المواد الغذائية total revenue | Comparison EN | Table + Chart |
| 127 | قارن متوسط قيمة الطلب بين فئة المنظفات والمناديل | Comparison AR | Table + Chart |
| 128 | Compare revenue of done orders vs invoiced orders | Comparison EN | Table + Chart |
| 129 | مقارنة إيرادات الشهر الحالي بالشهر السابق | Comparison AR | Table + Chart |
| 130 | Compare January 2025 revenue vs January 2024 | Comparison EN | Table + Chart |
| 131 | قارن أعلى 5 منتجات بين 2024 و 2025 | Compound AR | Table + Chart |
| 132 | Compare brand TIDE vs ARIEL across all years | Comparison EN | Table + Chart |
| 133 | مقارنة الكمية المباعة بين الربع الأول والثاني في 2025 | Comparison AR | Table + Chart |
| 134 | Compare discount amount given in 2024 vs 2025 | Comparison EN | Table + Chart |
| 135 | قارن إيرادات يناير ويونيو 2025 | Comparison AR | Table + Chart |
| 136 | Compare top category revenue between first and second half of 2025 | Comparison EN | Table + Chart |
| 137 | مقارنة متوسط سعر الوحدة بين الفئات | Comparison AR | Table + HBar |
| 138 | Compare number of products sold per category 2024 vs 2025 | Comparison EN | Table + Chart |
| 139 | قارن إيرادات الطلبات ذات الخصم وبدون خصم | Comparison AR | Table + Chart |
| 140 | Compare average tax per category | Comparison EN | Table + HBar |

---

## 8. Compound / Multi-part Queries (25 queries)

| #  | Query | Type | Expected |
|----|-------|------|----------|
| 141 | أكثر 5 منتجات إيرادات واتجاه مبيعاتهم الشهري في 2025 | Compound AR | 2 Tables + Chart |
| 142 | Top 5 categories by revenue and their monthly trend in 2024 | Compound EN | 2 Tables + Chart |
| 143 | أكثر 5 فئات وأكثر 5 علامات تجارية إيرادات في 2025 | Compound AR | 2 Tables + Chart |
| 144 | Top 5 products by revenue and top 5 by quantity sold | Compound EN | 2 Tables + Chart |
| 145 | إيرادات كل فئة في 2024 ونموها في 2025 | Compound AR | 2 Tables + Chart |
| 146 | Revenue by category in 2024 and growth percentage to 2025 | Compound EN | 2 Tables + Chart |
| 147 | أكثر 3 منتجات في فئة المنظفات واتجاه مبيعاتهم | Compound AR | 2 Tables + Chart |
| 148 | Top brands in المواد الغذائية and their monthly revenue 2025 | Compound EN | 2 Tables + Chart |
| 149 | قارن أعلى فئة وأدنى فئة من حيث الإيرادات | Compound AR | 2 Tables + Chart |
| 150 | Top 5 products by revenue and bottom 5 by revenue | Compound EN | 2 Tables + Chart |
| 151 | أكثر 5 منتجات وتطور مبيعات كل منهم شهرياً | Compound AR | 2 Tables + Chart |
| 152 | Monthly trend for المنظفات and العناية الشخصية side by side | Compound EN | 2 Tables + Chart |
| 153 | إجمالي الإيرادات والطلبات المكتملة لكل فئة | Compound AR | Table |
| 154 | Revenue and order count per brand for top 10 | Compound EN | Table + Chart |
| 155 | أكثر 5 منتجات إيرادات في 2024 وهل نمت في 2025؟ | Compound AR | 2 Tables + Chart |
| 156 | Categories with highest revenue and their best-selling product | Compound EN | 2 Tables |
| 157 | أكثر 3 فئات إيرادات وأكثر 3 علامات تجارية في 2025 | Compound AR | 2 Tables + Chart |
| 158 | Monthly revenue for top 3 categories in 2025 | Compound EN | Table + Chart |
| 159 | قارن مبيعات يناير وفبراير ومارس 2025 لكل فئة | Compound AR | Table + Chart |
| 160 | Top 5 products by revenue and discount received | Compound EN | 2 Tables + Chart |
| 161 | أكثر 10 منتجات وأكثر 10 فئات من حيث الكميات المباعة | Compound AR | 2 Tables + Chart |
| 162 | Revenue and quantity sold for PAMPERS in 2024 and 2025 | Compound EN | Table + Chart |
| 163 | أكثر 5 منتجات وأكثر 5 فئات إيرادات في الربع الأول 2025 | Compound AR | 2 Tables + Chart |
| 164 | Top selling products in done orders vs waiting orders | Compound EN | 2 Tables + Chart |
| 165 | إيرادات أكثر 3 فئات شهرياً مع إجمالي الخصومات | Compound AR | 2 Tables + Chart |

---

## 9. Chart Edit (Live Modification) (15 queries)

> Run these AFTER a chart has been generated by a previous query.

| #  | Trigger Query | Chart Edit Command | Expected |
|----|--------------|-------------------|----------|
| 166 | (after ranking chart) | اقلب المخطط إلى عمودي | Chart updated to vertical bar |
| 167 | (after bar chart) | flip to horizontal bar | Chart updated to hbar |
| 168 | (after any chart) | غيّر لون الأعمدة إلى أخضر | Chart color changed |
| 169 | (after any chart) | change chart color to red | Chart color changed |
| 170 | (after line chart) | حوّل المخطط إلى مخطط مساحي | Chart type changed to area |
| 171 | (after bar chart) | convert to pie chart | Chart type changed to pie |
| 172 | (after any chart) | رتب البيانات تنازلياً | Chart data sorted descending |
| 173 | (after any chart) | sort bars ascending | Chart sorted ascending |
| 174 | (after any chart) | أضف عنواناً للمخطط: الإيرادات الشهرية | Chart title updated |
| 175 | (after any chart) | add title: Revenue Analysis 2025 | Chart title added |
| 176 | (after bar chart) | اجعل المخطط أفقياً | Chart flipped to hbar |
| 177 | (after line chart) | show data points as markers only | Chart style changed |
| 178 | (after any chart) | غيّر الخلفية إلى داكنة | Chart theme changed |
| 179 | (after any chart) | make chart background white | Background changed |
| 180 | (after bar chart) | اعكس ترتيب المحور الأفقي | Axis reversed |

---

## 10. Semantic Cache Tests (10 queries)

> Pairs — send query A, then query B. B should hit the semantic cache.

| #  | Query A (store) | Query B (should HIT cache) | Cache? |
|----|----------------|---------------------------|--------|
| 181 | أكثر 5 منتجات إيرادات | ما هي أعلى 5 منتجات تحقيقاً للمبيعات؟ | ✅ HIT |
| 182 | Top 5 products by revenue | What are the five best-selling products by revenue? | ✅ HIT |
| 183 | إجمالي الإيرادات في 2024 | ما مجموع المبيعات خلال عام 2024؟ | ✅ HIT |
| 184 | Monthly revenue trend 2025 | Show me monthly sales trend for year 2025 | ✅ HIT |
| 185 | أكثر 10 فئات إيرادات | What are the top 10 categories by revenue? | ❌ MISS (lang differs) |
| 186 | Top 5 products by revenue | Top 10 products by revenue | ❌ MISS (top_n differs) |
| 187 | أكثر 5 منتجات إيرادات في 2024 | أكثر 5 منتجات إيرادات في 2025 | ❌ MISS (time differs) |
| 188 | أكثر 5 منتجات إيرادات | أكثر 5 فئات إيرادات | ❌ MISS (dimension differs) |
| 189 | إيرادات شهر يناير 2025 | مبيعات يناير 2025 | ✅ HIT |
| 190 | Revenue by category 2025 | Sales breakdown by category in 2025 | ✅ HIT |

---

## 11. Chitchat Gate Tests (5 queries)

| #  | Query | Expected |
|----|-------|----------|
| 191 | مرحباً كيف حالك؟ | Friendly chitchat response |
| 192 | What is the capital of Kuwait? | Friendly chitchat response |
| 193 | من أنت؟ ما قدراتك؟ | Agent intro response |
| 194 | اشرح لي ما هو التعلم الآلي | Friendly chitchat response |
| 195 | Tell me a joke | Friendly chitchat response |

---

## 12. Reference Resolution / Follow-up Tests (10 queries)

> Send these as follow-ups immediately after the query in the "After" column.

| #  | After query | Follow-up | Expected |
|----|------------|-----------|----------|
| 196 | أكثر 10 منتجات إيرادات | والان فقط في 2024 | Same query filtered to 2024 |
| 197 | Top 5 categories by revenue | same but for 2025 only | Same query for 2025 |
| 198 | أكثر 5 علامات تجارية إيرادات | والان أكثر 10 | Same query with top_n=10 |
| 199 | Monthly revenue trend in 2024 | now show the same for 2025 | Same trend for 2025 |
| 200 | إيرادات فئة المنظفات في 2024 | وماذا عن فئة المواد الغذائية؟ | Same metric, different category |

---

## Quick Reference — Test Coverage Matrix

| Capability | Queries |
|---|---|
| Arabic queries | 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,96,98,100,102,104,106,108,110,112,114,116,118,120,121,123,125,127,129,131,133,135,137,139,141,143,145,147,149,151,153,155,157,159,161,163,165,166,168,170,172,174,176,178,180,181,183,185,187,189,191,193,196,198,200 |
| English queries | 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,97,99,101,103,105,107,109,111,113,115,117,119,122,124,126,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,167,169,171,173,175,177,179,182,184,186,188,190,192,194,197,199 |
| KPI (single value) | 1–20, 45,46,67,68,83–88,106,107,120 |
| Ranking | 21–24,27,28,32,35,40,43,44,53,61,62,69,70,73,76,78,86 |
| Trend / Time-series | 47,48,63,64,89,95–120 |
| Comparison | 49,50,65,66,90,94,121–140 |
| Compound | 38,39,112,113,141–165 |
| Chart Edit | 166–180 |
| Semantic Cache | 181–190 |
| Chitchat Gate | 191–195 |
| Reference Resolution | 196–200 |
