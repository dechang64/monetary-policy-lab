"""Home / Landing page."""
import streamlit as st

def render():
    st.markdown(
        '<div class="main-header"><h1>💰 Personal Finance Teaching Lab</h1>'
        '<p>10 interactive modules for classroom use — built with Streamlit</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown("## 📋 Module Overview")

    cols = st.columns(4)
    modules = [
        ("📊", "Budget Planner", "Income/expense tracking and visualization"),
        ("✨", "Compound Interest", "See how money grows over 30+ years"),
        ("👴", "Retirement Planner", "Monte Carlo simulation for retirement"),
        ("🏠", "Buy vs Rent", "NPV comparison of buying vs renting"),
        ("📈", "Portfolio Optimization", "Markowitz efficient frontier"),
        ("💳", "Credit Card Payoff", "True cost of minimum payments"),
        ("🧾", "Tax Calculator", "Federal + state income tax"),
        ("🎓", "Student Loan Planner", "Standard vs income-driven repayment"),
        ("🏆", "FICO Score Simulator", "Interactive credit score factors"),
        ("🛡️", "Insurance Needs", "Life insurance coverage estimation"),
    ]
    for i, (icon, name, desc) in enumerate(modules):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{icon} {name}</h3>
                <p style="font-size:0.85rem; font-weight:400; color:#64748b;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎓 How to Use")
    st.markdown("""
    1. **In class**: Open a module, project to students, walk through the concept
    2. **Homework**: Students adjust sliders, export results as PDF (browser print)
    3. **Discussion**: Each module has a "Discussion Questions" section at the bottom

    **Tip**: All calculations run in real-time. Change any input and see results update instantly.
    """)

    st.markdown("---")
    st.markdown("## 📚 Teaching Philosophy")
    st.info("""
    **Core principle**: Students learn personal finance by doing, not memorizing.

    Each module is designed around a **single financial decision** that students
    will face in real life. The interactive tools let them explore trade-offs
    and develop intuition.

    **Suggested course flow**:
    - Week 1-2: Budget + Compound Interest (foundations)
    - Week 3-4: Credit Card + FICO (credit fundamentals)
    - Week 5-6: Student Loan + Tax (real-world obligations)
    - Week 7-8: Buy vs Rent + Portfolio (investment decisions)
    - Week 9-10: Retirement + Insurance (long-term planning)
    """)
