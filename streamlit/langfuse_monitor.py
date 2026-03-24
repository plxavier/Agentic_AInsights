# streamlit_monitor_fixed.py - CORRECT COST RETRIEVAL
import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Langfuse Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Langfuse Monitoring Dashboard")
st.caption("Track token usage, costs, latency, and model performance")


def get_langfuse_client():
    try:
        from langfuse import Langfuse
        from dotenv import load_dotenv
        load_dotenv()

        return Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
    except Exception as e:
        st.error(f"Langfuse error: {e}")
        return None


@st.cache_data(ttl=None)
def fetch_detailed_traces(hours=24):
    langfuse = get_langfuse_client()
    if not langfuse:
        return [], []

    try:
        traces = langfuse.fetch_traces(
            limit=50,
            from_timestamp=datetime.now() - timedelta(hours=hours)
        )

        trace_data = []
        all_observations = []

        for trace in traces.data:
            total_input = 0
            total_output = 0
            total_cost = 0
            total_latency = 0
            obs_count = 0
            model_stats = {}

            try:
                observations = langfuse.fetch_observations(trace_id=trace.id)
                for obs in observations.data:
                    # Get usage data - try different possible attribute names
                    usage = None
                    if hasattr(obs, 'usage') and obs.usage:
                        usage = obs.usage
                    elif hasattr(obs, 'usage_details') and obs.usage_details:
                        usage = obs.usage_details

                    if usage:
                        # Try different ways to get token counts
                        input_t = getattr(usage, 'input', 0) or getattr(usage, 'prompt_tokens', 0) or 0
                        output_t = getattr(usage, 'output', 0) or getattr(usage, 'completion_tokens', 0) or 0
                        cost = getattr(usage, 'total_cost', 0) or 0

                        # If cost is still 0, try to calculate from tokens
                        if cost == 0 and (input_t > 0 or output_t > 0):
                            # Approximate cost for GPT-4.1-mini
                            input_cost = (input_t / 1_000_000) * 0.15
                            output_cost = (output_t / 1_000_000) * 0.60
                            cost = input_cost + output_cost

                        total_input += input_t
                        total_output += output_t
                        total_cost += cost
                        obs_count += 1

                        # Get latency
                        latency = 0
                        if hasattr(obs, 'latency') and obs.latency:
                            latency = obs.latency
                        elif hasattr(obs, 'end_time') and hasattr(obs, 'start_time'):
                            if obs.end_time and obs.start_time:
                                latency = (obs.end_time - obs.start_time).total_seconds()
                        total_latency += latency

                        # Model name
                        model = getattr(obs, 'model', 'unknown')
                        if model not in model_stats:
                            model_stats[model] = {'input': 0, 'output': 0, 'cost': 0, 'calls': 0}
                        model_stats[model]['input'] += input_t
                        model_stats[model]['output'] += output_t
                        model_stats[model]['cost'] += cost
                        model_stats[model]['calls'] += 1

                        all_observations.append({
                            'trace_id': trace.id[:16],
                            'trace_name': trace.name or 'research_session',
                            'timestamp': obs.start_time or trace.timestamp,
                            'model': model,
                            'input_tokens': input_t,
                            'output_tokens': output_t,
                            'total_tokens': input_t + output_t,
                            'cost': cost,
                            'latency': latency,
                            'observation_name': getattr(obs, 'name', 'unknown')
                        })
            except Exception as e:
                st.warning(f"Error fetching observations: {e}")

            trace_data.append({
                'trace_id': trace.id[:16],
                'timestamp': trace.timestamp,
                'name': trace.name or 'research_session',
                'input_tokens': total_input,
                'output_tokens': total_output,
                'total_tokens': total_input + total_output,
                'cost': total_cost,
                'latency': total_latency,
                'observations': obs_count,
                'models': model_stats
            })

        return trace_data, all_observations
    except Exception as e:
        st.warning(f"Could not fetch traces: {e}")
        return [], []


# Sidebar
with st.sidebar:
    st.header("🎛️ Controls")
    hours = st.slider("Time Range (hours)", min_value=1, max_value=168, value=24)

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.header("📊 Display Options")
    show_observations = st.checkbox("Show Detailed Observations", value=False)

    st.markdown("---")
    st.header("🔗 Direct Links")
    st.markdown("[Open Langfuse Dashboard](https://cloud.langfuse.com)")
    st.markdown("[Your Project](https://cloud.langfuse.com/project/cmn0s4egt00ewad07ck5trmwm)")

    st.markdown("---")
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# Fetch data
with st.spinner("Fetching trace data..."):
    traces, observations = fetch_detailed_traces(hours)

# Summary Metrics
st.subheader("📊 Summary")
col1, col2, col3, col4, col5 = st.columns(5)

total_cost = sum(t['cost'] for t in traces)
total_tokens = sum(t['total_tokens'] for t in traces)
total_sessions = len(traces)
total_llm_calls = sum(t['observations'] for t in traces)
avg_latency = sum(t['latency'] for t in traces) / total_llm_calls if total_llm_calls > 0 else 0

with col1:
    st.metric("📊 Sessions", total_sessions)
with col2:
    st.metric("🔄 LLM Calls", total_llm_calls)
with col3:
    st.metric("📝 Total Tokens", f"{total_tokens:,}")
with col4:
    st.metric("💰 Total Cost", f"${total_cost:.4f}")
with col5:
    st.metric("⏱️ Avg Latency", f"{avg_latency:.2f}s")

st.markdown("---")

if traces:
    # 1. Cost per Session - FIXED with debug
    st.subheader("💰 Cost per Session")

    # Debug: Show raw cost data
    with st.expander("Debug: Raw Cost Data"):
        st.write("Cost values from traces:")
        for t in traces:
            st.write(f"- {t['name']}: ${t['cost']:.6f} (tokens: {t['total_tokens']})")

    # Filter traces with positive cost
    cost_data = []
    for t in traces:
        if t['cost'] > 0:
            cost_data.append({
                'Session': t['name'][:25],
                'Cost': t['cost'],
                'Timestamp': t['timestamp'],
                'Tokens': t['total_tokens']
            })

    if cost_data:
        cost_df = pd.DataFrame(cost_data).sort_values('Timestamp')

        # Create bar chart
        fig = px.bar(cost_df, x='Session', y='Cost',
                     title='Cost per Research Session',
                     color='Cost',
                     color_continuous_scale='Greens',
                     text='Cost',
                     hover_data={'Tokens': True})

        fig.update_traces(texttemplate='$%{text:.4f}', textposition='outside')
        fig.update_layout(yaxis_tickformat='$.4f')
        st.plotly_chart(fig, use_container_width=True)

        # Cost details table
        st.subheader("💰 Cost Details")
        cost_detail = pd.DataFrame([{
            'Session': t['name'][:25],
            'Time': t['timestamp'].strftime('%Y-%m-%d %H:%M') if t['timestamp'] else 'N/A',
            'Input Tokens': f"{t['input_tokens']:,}",
            'Output Tokens': f"{t['output_tokens']:,}",
            'Total Tokens': f"{t['total_tokens']:,}",
            'Cost': f"${t['cost']:.4f}"
        } for t in traces if t['cost'] > 0])
        st.dataframe(cost_detail, use_container_width=True)
    else:
        st.info(f"No cost data found for the last {hours} hours. Cost values: {[t['cost'] for t in traces]}")

    # 2. Tokens per Session
    st.subheader("📊 Tokens per Session")
    token_df = pd.DataFrame(
        [{'Session': t['name'][:25], 'Input': t['input_tokens'], 'Output': t['output_tokens']} for t in traces])
    if not token_df.empty and token_df['Input'].sum() > 0:
        fig = go.Figure(data=[
            go.Bar(name='Input', x=token_df['Session'], y=token_df['Input'], marker_color='#3498db'),
            go.Bar(name='Output', x=token_df['Session'], y=token_df['Output'], marker_color='#e74c3c')
        ])
        fig.update_layout(title='Input vs Output Tokens per Session', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # 3. Token Usage Over Time
    st.subheader("📈 Token Usage Over Time")
    time_df = pd.DataFrame([{'timestamp': t['timestamp'], 'tokens': t['total_tokens']} for t in traces if
                            t['total_tokens'] > 0]).sort_values('timestamp')
    if not time_df.empty:
        fig = px.line(time_df, x='timestamp', y='tokens', title='Token Usage Trend', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # 4. Latency per Session
    st.subheader("⏱️ Latency per Session")
    latency_df = pd.DataFrame(
        [{'Session': t['name'][:25], 'Latency (s)': t['latency']} for t in traces if t['latency'] > 0])
    if not latency_df.empty:
        fig = px.bar(latency_df, x='Session', y='Latency (s)', title='Total Latency per Session',
                     color='Latency (s)', color_continuous_scale='Oranges')
        st.plotly_chart(fig, use_container_width=True)

    # 5. Model Usage
    st.subheader("🤖 Model Usage")
    all_models = {}
    for t in traces:
        for model, stats in t['models'].items():
            if model not in all_models:
                all_models[model] = {'calls': 0, 'cost': 0}
            all_models[model]['calls'] += stats['calls']
            all_models[model]['cost'] += stats['cost']

    if all_models:
        model_df = pd.DataFrame([{'Model': m, 'Calls': s['calls'], 'Cost': s['cost']} for m, s in all_models.items()])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(model_df, values='Calls', names='Model', title='Calls by Model')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if model_df['Cost'].sum() > 0:
                fig = px.pie(model_df, values='Cost', names='Model', title='Cost by Model')
                st.plotly_chart(fig, use_container_width=True)

    # 6. Sessions Table
    st.subheader(f"📋 Recent Sessions (Last {hours} hours)")
    display_df = pd.DataFrame([{
        'Time': t['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if t['timestamp'] else 'N/A',
        'Session': t['name'],
        'Calls': t['observations'],
        'Input': f"{t['input_tokens']:,}",
        'Output': f"{t['output_tokens']:,}",
        'Total': f"{t['total_tokens']:,}",
        'Latency': f"{t['latency']:.1f}s",
        'Cost': f"${t['cost']:.4f}",
        'Trace ID': t['trace_id']
    } for t in traces])
    st.dataframe(display_df, use_container_width=True)

    # 7. Links
    st.markdown("### 🔗 View in Langfuse")
    for t in traces[:5]:
        trace_url = f"https://cloud.langfuse.com/trace/{t['trace_id']}"
        st.markdown(f"- [{t['name']}]({trace_url}) - ${t['cost']:.4f} - {t['latency']:.1f}s")

    # 8. Detailed Observations
    if show_observations and observations:
        st.markdown("---")
        st.subheader("🔬 Detailed Observations")
        obs_df = pd.DataFrame(observations[-50:]).sort_values('timestamp', ascending=False)
        if not obs_df.empty:
            st.dataframe(obs_df[['trace_name', 'model', 'input_tokens', 'output_tokens',
                                 'total_tokens', 'latency', 'cost', 'timestamp']].head(20),
                         use_container_width=True)

else:
    st.info("No traces found. Complete a research session to see data.")

# Cost Calculator
st.markdown("---")
st.subheader("💰 Cost Calculator")
col1, col2 = st.columns(2)
with col1:
    est_sessions = st.number_input("Sessions", min_value=1, value=1)
    est_llm_calls = st.number_input("Calls/Session", min_value=1, value=8)
    est_input = st.number_input("Input Tokens/Call", value=500)
    est_output = st.number_input("Output Tokens/Call", value=1500)

with col2:
    total_calls = est_sessions * est_llm_calls
    total_input = total_calls * est_input
    total_output = total_calls * est_output
    input_cost = (total_input / 1_000_000) * 0.15
    output_cost = (total_output / 1_000_000) * 0.60
    st.metric("Estimated Cost", f"${input_cost + output_cost:.4f}")
    st.caption(f"{total_calls} calls | {total_input:,} in | {total_output:,} out")

st.markdown("---")
st.caption("Data refreshes every 60 seconds | Source: Langfuse API")