"""Budget Planner — income/expense tracking and visualization."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render():
    st.markdown('<div class="main-header"><h1>📊 Budget Planner</h1>'
                '<p>Track income and expenses — understand cash flow</p></div>',
                unsafe_allow_html=True)

    st.markdown("### 💰 Income")
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_income = st.number_input("Monthly Take-Home Pay ($)", 0, 100000, 5000, step=500)
    with col2:
        side_income = st.number_input("Side Income ($/month)", 0, 50000, 0, step=200)
    with col3:
        annual_bonus = st.number_input("Annual Bonus ($)", 0, 200000, 0, step=1000)

    st.markdown("### 📝 Monthly Expenses")
    expense_categories = [
        ("Housing (rent/mortgage)", 1800),
        ("Food & Groceries", 600),
        ("Transportation", 400),
        ("Utilities & Internet", 250),
        ("Phone & Subscriptions", 120),
        ("Health & Insurance", 300),
        ("Entertainment & Dining", 400),
        ("Personal Care", 100),
        ("Savings & Investment", 500),
        ("Other", 200),
    ]
    expenses = {}
    cols = st.columns(2)
    for i, (cat, default) in enumerate(expense_categories):
        with cols[i % 2]:
            expenses[cat] = st.number_input(f"{cat} ($)", 0, 20000, default, step=50,
                                              key=f"exp_{cat}")

    total_monthly_income = monthly_income + side_income
    total_monthly_expenses = sum(expenses.values())
    monthly_savings = total_monthly_income - total_monthly_expenses
    savings_rate = (monthly_savings / total_monthly_income * 100) if total_monthly_income > 0 else 0

    st.markdown("---")
    st.markdown("### 📊 Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Income", f"${total_monthly_income:,.0f}")
    col2.metric("Monthly Expenses", f"${total_monthly_expenses:,.0f}")
    col3.metric("Monthly Savings", f"${monthly_savings:,.0f}",
                delta=f"{savings_rate:.1f}% rate" if monthly_savings >= 0 else "Deficit!",
                delta_color="normal" if monthly_savings >= 0 else "inverse")
    col4.metric("Annual Savings", f"${monthly_savings * 12 + annual_bonus:,.0f}")

    # Visualization
    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure(data=[go.Pie(
            labels=list(expenses.keys()),
            values=list(expenses.values()),
            hole=.4,
            textinfo='label+percent',
        )])
        fig.update_layout(title="Expense Breakdown", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Sankey-style flow
        fig2 = go.Figure(data=[go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=["relative", "relative"] + ["relative"]*len(expenses) + ["total"],
            x=["Income", "Side Income"] + list(expenses.keys()) + ["Net"],
            y=[monthly_income, side_income] + [-v for v in expenses.values()] + [0],
            connector={"line":{"color":"#94a3b8"}},
            increasing={"marker":{"color":"#16a34a"}},
            decreasing={"marker":{"color":"#dc2626"}},
            totals={"marker":{"color":"#3b82f6"}},
        )])
        fig2.update_layout(title="Monthly Cash Flow Waterfall", height=400, yaxis_title="$")
        st.plotly_chart(fig2, use_container_width=True)

    # 50/30/20 Rule
    st.markdown("### 📐 50/30/20 Rule Check")
    needs_cats = ["Housing (rent/mortgage)", "Food & Groceries", "Transportation",
                  "Utilities & Internet", "Health & Insurance"]
    wants_cats = ["Entertainment & Dining", "Personal Care", "Phone & Subscriptions"]
    savings_cats = ["Savings & Investment"]

    needs = sum(expenses.get(c, 0) for c in needs_cats)
    wants = sum(expenses.get(c, 0) for c in wants_cats)
    savings = sum(expenses.get(c, 0) for c in savings_cats)
    other = expenses.get("Other", 0)

    n_total = needs + wants + savings + other
    if n_total > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Needs (target 50%)", f"${needs:,.0f}", f"{needs/n_total*100:.0f}%")
        col2.metric("Wants (target 30%)", f"${wants:,.0f}", f"{wants/n_total*100:.0f}%")
        col3.metric("Savings (target 20%)", f"${savings:,.0f}", f"{savings/n_total*100:.0f}%")

    st.markdown("""
    **Discussion Questions:**
    1. Is your savings rate above 20%? What would you cut to reach 30%?
    2. What percentage goes to "needs" vs "wants"? Is it sustainable?
    3. If your income dropped 20%, which expenses would you cut first?
    """)
