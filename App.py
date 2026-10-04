import os
import streamlit as st
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Streamlit Page Config (Office Dashboard UI)
st.set_page_config(page_title="Trading Desk AI - Multi-Agent System", layout="wide")

st.title("📈 Trading Research Office - AI Multi-Agent Desk")
st.markdown("---")

# Sidebar for Setup & API Configuration
with st.sidebar:
    st.header("⚙️ Office Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    ticker = st.text_input("Stock/Crypto Ticker (e.g., RELIANCE.NS, AAPL, BTC-USD)", "RELIANCE.NS").upper()
    timeframe = st.selectbox("Primary Analysis Focus", ["Intraday / Short-Term", "Swing Trading", "Long-Term Investment"])
    run_btn = st.button("🚀 Start Team Analysis")

if run_btn:
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar!")
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

        # Fetch Data
        st.info(f"📊 Fetching market data for {ticker}...")
        stock_data = yf.Ticker(ticker)
        hist = stock_data.history(period="6mo")
        info = stock_data.info

        data_summary = f"""
        Ticker: {ticker}
        Current Price: {info.get('currentPrice', hist['Close'].iloc[-1])}
        52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
        52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
        P/E Ratio: {info.get('trailingPE', 'N/A')}
        Market Cap: {info.get('marketCap', 'N/A')}
        Recent 5-day Prices: {hist['Close'].tail().tolist()}
        """

        # -------------------------------------------------------------
        # AGENTS DEFINITION (Office Team)
        # -------------------------------------------------------------

        # 1. Technical Specialist
        tech_agent = Agent(
            role="Senior Technical Analyst",
            goal="Analyze price action, support/resistance, RSI, MACD, and trendlines.",
            backstory="You are a veteran technical chartist with 15 years of experience in technical patterns and chart indicators.",
            llm=llm,
            verbose=True
        )

        # 2. Multi-Timeframe Specialist
        timeframe_agent = Agent(
            role="Multi-Timeframe Strategist",
            goal="Analyze trend consistency across daily, weekly, and monthly timeframes.",
            backstory="You specialize in finding high-probability setups where multiple timeframes align together.",
            llm=llm,
            verbose=True
        )

        # 3. Short-Term / Scalp Specialist
        scalp_agent = Agent(
            role="Short-Term Execution Specialist",
            goal="Evaluate immediate momentum, order flow, and short-term volatility for quick entries.",
            backstory="You focus on intraday price swings, breakout confirmations, and quick target opportunities.",
            llm=llm,
            verbose=True
        )

        # 4. Long-Term / Fundamental Specialist
        fundamental_agent = Agent(
            role="Chief Fundamental Analyst",
            goal="Analyze business metrics, valuation, earnings growth, and long-term viability.",
            backstory="You evaluate macro trends, valuation ratios, and financial health to judge long-term investments.",
            llm=llm,
            verbose=True
        )

        # 5. Risk & Sentiment Manager
        risk_agent = Agent(
            role="Risk Management Officer",
            goal="Calculate exact Risk-to-Reward ratio, Stop Loss level, and Market Sentiment risk.",
            backstory="You protect capital at all costs and ensure trades do not over-expose the portfolio to high risk.",
            llm=llm,
            verbose=True
        )

        # 6. SUPERVISOR AGENT
        supervisor_agent = Agent(
            role="Chief Investment Officer (Supervisor)",
            goal="Supervise all 5 agents, consolidate their analysis, resolve conflicting views, and produce a unified executive trading report.",
            backstory="You are the Head of Research. You review reports from all specialists, verify their findings, and write the final actionable report for the trader.",
            llm=llm,
            verbose=True
        )

        # -------------------------------------------------------------
        # TASKS ASSIGNMENT
        # -------------------------------------------------------------

        t1 = Task(description=f"Analyze technical charts & price levels for {ticker} based on data: {data_summary}", agent=tech_agent, expected_output="Technical analysis report.")
        t2 = Task(description=f"Analyze multi-timeframe alignment for {ticker} for {timeframe}.", agent=timeframe_agent, expected_output="Multi-timeframe trend report.")
        t3 = Task(description=f"Assess short-term breakout/entry opportunities for {ticker}.", agent=scalp_agent, expected_output="Short-term entry/exit report.")
        t4 = Task(description=f"Assess long-term valuation and fundamental strength for {ticker}.", agent=fundamental_agent, expected_output="Fundamental report.")
        t5 = Task(description=f"Calculate exact Stop-Loss, Target levels, and Risk/Reward for {ticker}.", agent=risk_agent, expected_output="Risk assessment report.")

        t_supervisor = Task(
            description=f"""
            Collect and review outputs from all 5 specialists.
            Synthesize them into a single, clean executive Office Research Report for {ticker}.
            The report MUST include:
            1. Executive Summary & Final Verdict (Bullish / Bearish / Neutral)
            2. Short-Term Strategy (Entry, Target, Stop Loss)
            3. Long-Term Strategy & Fundamental Health
            4. Risk Score (1 to 10) & Position Sizing Recommendation
            """,
            agent=supervisor_agent,
            expected_output="Final Comprehensive Executive Trading Report."
        )

        # -------------------------------------------------------------
        # CREW EXECUTION
        # -------------------------------------------------------------

        trading_office = Crew(
            agents=[tech_agent, timeframe_agent, scalp_agent, fundamental_agent, risk_agent, supervisor_agent],
            tasks=[t1, t2, t3, t4, t5, t_supervisor],
            process=Process.sequential,
            verbose=True
        )

        with st.spinner("🤖 Office Team is analyzing the stock... Please wait..."):
            result = trading_office.kickoff()

        # Display Final Executive Output
        st.success("✅ Team Analysis Completed!")
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Executive Office Report (From Chief Analyst)")
            st.markdown(result)

        with col2:
            st.subheader("📌 Stock Summary Data")
            st.json({
                "Ticker": ticker,
                "Price": info.get('currentPrice', hist['Close'].iloc[-1]),
                "52W High": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52W Low": info.get('fiftyTwoWeekLow', 'N/A'),
                "P/E Ratio": info.get('trailingPE', 'N/A')
            })
            st.line_chart(hist['Close'])
