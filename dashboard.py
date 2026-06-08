import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def fraud_overview(df: pd.DataFrame) -> dict:
    total      = len(df)
    fraud_cnt  = int(df["is_fraud"].sum())
    legit_cnt  = total - fraud_cnt
    fraud_pct  = fraud_cnt / total * 100
    total_amt  = df["amount"].sum()
    fraud_amt  = df[df["is_fraud"] == 1]["amount"].sum()
    return dict(total=total, fraud=fraud_cnt, legit=legit_cnt,
                fraud_pct=fraud_pct, total_amt=total_amt, fraud_amt=fraud_amt)


def fraud_by_bank(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return (
        df[df["is_fraud"] == 1]
        .groupby("sender_bank")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="fraud_count")
    )


def fraud_by_state(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["is_fraud"] == 1]
        .groupby("sender_state")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="fraud_count")
    )


def fraud_by_method(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("payment_method")["is_fraud"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud", "count": "total"})
        .assign(fraud_rate=lambda x: x["fraud"] / x["total"] * 100)
        .reset_index()
    )


def fraud_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour")["is_fraud"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "fraud", "count": "total"})
        .reset_index()
    )


def plot_fraud_donut(overview: dict) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Legit", "Fraud"],
        values=[overview["legit"], overview["fraud"]],
        hole=0.55,
        marker_colors=["#2ecc71", "#e74c3c"]
    ))
    fig.update_layout(showlegend=True, margin=dict(t=20, b=20))
    return fig


def plot_bank_bar(bank_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(bank_df, x="sender_bank", y="fraud_count",
                 color="fraud_count", color_continuous_scale="Reds",
                 labels={"sender_bank": "Bank", "fraud_count": "Fraud Count"})
    fig.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
    return fig


def plot_hourly(hour_df: pd.DataFrame) -> go.Figure:
    fig = px.line(hour_df, x="hour", y="fraud",
                  markers=True, labels={"hour": "Hour", "fraud": "Fraud Count"})
    fig.update_traces(line_color="#e74c3c")
    fig.update_layout(margin=dict(t=20, b=20))
    return fig


def plot_state_map(state_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(state_df, x="sender_state", y="fraud_count",
                 color="fraud_count", color_continuous_scale="OrRd",
                 labels={"sender_state": "State", "fraud_count": "Fraud Count"})
    fig.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
    return fig


def plot_method_pie(method_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(method_df, names="payment_method", values="fraud",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(margin=dict(t=20, b=20))
    return fig
