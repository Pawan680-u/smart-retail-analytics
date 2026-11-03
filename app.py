from io import BytesIO
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_loader import read_retail_file
from src.preprocessing import preprocess_transactions
from src.feature_engineering import daily_product_sales
from src.sales_forecasting import forecast_product
from src.customer_segmentation import build_rfm
from src.inventory_management import inventory_template, recommend_inventory

st.set_page_config(page_title="RETAIL INTELLIGENCE AI", page_icon="🛍️", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} .block-container{padding-top:1.5rem;max-width:1440px}
[data-testid="stMetric"]{background:#f3f7f7;padding:16px 18px;border:1px solid #e1ebea;border-radius:14px}
[data-testid="stSidebar"]{background:#f5f8f8}.hero{padding:24px;border-radius:18px;background:linear-gradient(120deg,#0f766e,#164e63);color:white;margin-bottom:18px}
.muted{color:#657579;font-size:.92rem}
</style>""", unsafe_allow_html=True)

PAGES = ["Dashboard Overview", "Sales Analytics", "Sales Forecasting", "Inventory Management", "Customer Segmentation", "Machine Learning Models", "Model Evaluation", "Dataset Management", "About Project"]
with st.sidebar:
    st.markdown("## 🛍️ RETAIL INTELLIGENCE AI")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    upload = st.file_uploader("Upload retail transactions", type=["csv", "xlsx"], help="Expected columns include InvoiceNo, StockCode, Quantity, InvoiceDate, and UnitPrice.")

@st.cache_data(show_spinner=False)
def load_and_clean(raw):
    return preprocess_transactions(raw)

if upload is not None:
    try:
        raw = read_retail_file(upload)
        sales, audit = load_and_clean(raw)
        st.session_state["sales"] = sales
        st.session_state["audit"] = audit
        st.session_state["source_name"] = upload.name
    except Exception as exc:
        st.error(f"Could not load this dataset: {exc}")

sales = st.session_state.get("sales", pd.DataFrame())
audit = st.session_state.get("audit", {})
daily = daily_product_sales(sales) if not sales.empty else pd.DataFrame(columns=["StockCode", "Date", "Sales"])

def need_data():
    if sales.empty:
        st.info("Upload a CSV or Excel transaction file from the sidebar to populate this page. The app does not fabricate data or model metrics.")
        return False
    return True

def overview():
    st.markdown('<div class="hero"><h1>Retail performance, made actionable</h1><div>Sales intelligence, demand forecasts, customer insights, and explainable replenishment.</div></div>', unsafe_allow_html=True)
    if not need_data(): return
    orders = sales.InvoiceNo.nunique(); products = sales.StockCode.nunique(); customers = sales.CustomerID.nunique(); rev = sales.Revenue.sum()
    cols=st.columns(6)
    for c,(label,value) in zip(cols,[("Total Revenue",f"£{rev:,.2f}"),("Total Orders",f"{orders:,}"),("Total Products",f"{products:,}"),("Total Customers",f"{customers:,}"),("Avg Order Value",f"£{rev/orders:,.2f}" if orders else "£0"),("Units Sold",f"{sales.Quantity.sum():,.0f}")]): c.metric(label,value)
    monthly=sales.assign(Month=sales.InvoiceDate.dt.to_period("M").dt.to_timestamp()).groupby("Month",as_index=False).Revenue.sum()
    c1,c2=st.columns(2)
    c1.plotly_chart(px.line(monthly,x="Month",y="Revenue",title="Monthly Revenue Trend",template="plotly_white"),use_container_width=True)
    daily_rev=sales.assign(Date=sales.InvoiceDate.dt.floor("D")).groupby("Date",as_index=False).Revenue.sum()
    c2.plotly_chart(px.line(daily_rev,x="Date",y="Revenue",title="Daily Sales Trend",template="plotly_white"),use_container_width=True)
    products_top=sales.groupby("StockCode",as_index=False).Revenue.sum().nlargest(10,"Revenue")
    countries=sales.groupby("Country",as_index=False).Revenue.sum().nlargest(10,"Revenue")
    c1,c2=st.columns(2)
    c1.plotly_chart(px.bar(products_top,x="StockCode",y="Revenue",title="Top 10 Products",template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.bar(countries,x="Country",y="Revenue",title="Top Performing Countries",template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.histogram(sales,x="Revenue",nbins=40,title="Sales Distribution",template="plotly_white"),use_container_width=True)

def sales_analytics():
    st.title("Sales Analytics")
    if not need_data(): return
    a,b=st.columns(2)
    a.metric("Transactions",f"{len(sales):,}"); b.metric("Gross positive-sales revenue",f"£{sales.Revenue.sum():,.2f}")
    daily_rev=sales.assign(Date=sales.InvoiceDate.dt.floor("D")).groupby("Date",as_index=False).agg(Revenue=("Revenue","sum"),Units=("Quantity","sum"))
    measure=st.selectbox("Trend metric",["Revenue","Units"])
    st.plotly_chart(px.line(daily_rev,x="Date",y=measure,title=f"Daily {measure}",template="plotly_white"),use_container_width=True)
    st.subheader("Product performance")
    st.dataframe(sales.groupby("StockCode",as_index=False).agg(Units=("Quantity","sum"),Revenue=("Revenue","sum"),Orders=("InvoiceNo","nunique")).sort_values("Revenue",ascending=False),use_container_width=True)

def forecasting():
    st.title("Sales Forecasting")
    if not need_data(): return
    products=sorted(daily.StockCode.unique().astype(str)); product=st.selectbox("Product",products); horizon=st.selectbox("Forecast period",[7,15,30],format_func=lambda d:f"Next {d} days")
    if st.button("Generate forecast",type="primary"):
        with st.spinner("Evaluating models and generating a recursive forecast…"):
            try: result=forecast_product(daily,product,horizon)
            except Exception as exc: st.error(str(exc)); return
        if result["fallback"]: st.warning(f"Limited history for this product. Using {result['model_name']}; model comparison metrics are unavailable until more history is provided.")
        hist=result["history"].tail(180).rename(columns={"Sales":"Units"})[["Date","Units"]].assign(Series="Historical")
        future=result["forecast"].rename(columns={"PredictedSales":"Units"})[["Date","Units"]].assign(Series="Forecast")
        st.plotly_chart(px.line(pd.concat([hist,future]),x="Date",y="Units",color="Series",title=f"Product {product}: actual and forecast",template="plotly_white",color_discrete_map={"Historical":"#0f766e","Forecast":"#f59e0b"}),use_container_width=True)
        st.dataframe(result["forecast"],use_container_width=True)
        if result["metrics"]: st.subheader("Chronological holdout metrics"); st.dataframe(pd.DataFrame(result["metrics"]).T.rename_axis("Model"),use_container_width=True)
        if result["feature_importance"]: st.plotly_chart(px.bar(pd.DataFrame(result["feature_importance"].items(),columns=["Feature","Importance"]).sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Random Forest feature importance",template="plotly_white"),use_container_width=True)

def inventory_page():
    st.title("Inventory Management")
    if not need_data(): return
    st.caption("Stock, lead time, safety stock, and minimum order quantities come from your inventory input. Transaction data does not include live inventory.")
    tpl=inventory_template().to_csv(index=False).encode()
    st.download_button("Download inventory CSV template",tpl,"inventory_template.csv","text/csv")
    inv_file=st.file_uploader("Upload inventory CSV",type=["csv"],key="inventory_upload")
    if inv_file:
        try: inv=pd.read_csv(inv_file)
        except Exception as exc: st.error(f"Unable to read inventory file: {exc}"); return
    else:
        st.write("Enter inventory values for products to analyze.")
        inv=st.data_editor(pd.DataFrame({"StockCode":daily.StockCode.astype(str).drop_duplicates().head(10).tolist(),"CurrentStock":[0]*min(10,daily.StockCode.nunique()),"LeadTimeDays":[7]*min(10,daily.StockCode.nunique()),"SafetyStock":[0]*min(10,daily.StockCode.nunique()),"MinimumOrderQty":[0]*min(10,daily.StockCode.nunique())}),num_rows="dynamic",use_container_width=True,key="inventory_editor")
    try:
        # Estimate the daily rate from product forecasts; forecasting itself labels
        # sparse histories and falls back to a seasonal-naive or historical mean.
        forecast_rows=[]
        with st.spinner("Estimating product demand from the forecast models…"):
            for pid in inv.get("StockCode", pd.Series(dtype=str)).dropna().astype(str).unique():
                try:
                    projected=forecast_product(daily,pid,30)["forecast"]
                    forecast_rows.append(pd.DataFrame({"StockCode":pid,"Date":projected["Date"],"Sales":projected["PredictedSales"]}))
                except ValueError:
                    continue
        demand_estimates=pd.concat(forecast_rows,ignore_index=True) if forecast_rows else daily.iloc[0:0]
        result=recommend_inventory(inv,demand_estimates)
        st.dataframe(result,use_container_width=True)
        st.download_button("Download inventory recommendations",result.to_csv(index=False).encode(),"inventory_recommendations.csv","text/csv")
    except Exception as exc: st.error(f"Inventory values need attention: {exc}")

def customer_page():
    st.title("Customer Segmentation")
    if not need_data(): return
    rfm,info=build_rfm(sales)
    if rfm.empty: st.warning(info["message"]); return
    cols=st.columns(4)
    cols[0].metric("Identified customers",f"{len(rfm):,}"); cols[1].metric("Repeat customers",f"{int((rfm.Frequency>1).sum()):,}"); cols[2].metric("Avg lifetime spend",f"£{rfm.Monetary.mean():,.2f}"); cols[3].metric("Avg orders/customer",f"{rfm.Frequency.mean():.1f}")
    if info.get("silhouette") is not None: st.metric("Silhouette score",f"{info['silhouette']:.3f}")
    st.caption(info["message"])
    st.plotly_chart(px.scatter(rfm,x="Frequency",y="Monetary",color="Segment",size="Recency",hover_data=["CustomerID"],title="RFM customer groups",template="plotly_white"),use_container_width=True)
    st.dataframe(rfm.groupby("Segment",as_index=False).agg(Customers=("CustomerID","count"),AvgRecency=("Recency","mean"),AvgFrequency=("Frequency","mean"),AvgMonetary=("Monetary","mean")),use_container_width=True)

def evaluation_page():
    st.title("Model Evaluation")
    if not need_data(): return
    product=st.selectbox("Evaluate product",sorted(daily.StockCode.unique().astype(str)),key="eval_product")
    if st.button("Run chronological evaluation",type="primary"):
        with st.spinner("Training models on chronological data…"):
            result=forecast_product(daily,product,7)
        if result["metrics"]:
            table=pd.DataFrame(result["metrics"]).T.rename_axis("Model").reset_index()
            st.dataframe(table,use_container_width=True)
            st.plotly_chart(px.bar(table.melt(id_vars="Model",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error"),x="Model",y="Error",color="Metric",barmode="group",title="Holdout error comparison",template="plotly_white"),use_container_width=True)
        else: st.warning("Insufficient history for chronological ML evaluation; fallback forecasts are shown on Sales Forecasting.")

def dataset_page():
    st.title("Dataset Management")
    if not need_data(): return
    st.write(f"Loaded: **{st.session_state.get('source_name','uploaded dataset')}** · {len(sales):,} valid positive sales rows")
    st.subheader("Processing audit")
    metrics={k:v for k,v in audit.items() if not isinstance(v,pd.DataFrame)}
    st.json(metrics)
    st.caption("Cancelled invoices, negative quantity returns, and invalid rows are tracked separately and excluded from positive-sales demand forecasting. CustomerID may be missing; such rows remain in sales analytics but are excluded from RFM segmentation.")
    with st.expander("Preview cleaned sales"): st.dataframe(sales.head(100),use_container_width=True)
    if not audit.get("returns",pd.DataFrame()).empty:
        with st.expander("Return rows (separate from forecast demand)"): st.dataframe(audit["returns"].head(100),use_container_width=True)

def models_page():
    st.title("Machine Learning Models")
    st.write("Product demand models use lagged sales, rolling averages, weekday, and month. Validation and test observations follow time order; ensemble weights are based on validation MAE. Random forests expose feature importance.")
    if not need_data(): return
    st.metric("Products with daily sales history",f"{daily.StockCode.nunique():,}")
    st.info("Select a product on Model Evaluation to fit and compare Linear Regression, Decision Tree, Random Forest, and the validation-weighted ensemble. Forecasting uses recursive predictions because future actual lags are unavailable.")

def about():
    st.title("About Project")
    st.markdown("""**AI-Based Smart Retail Sales Prediction and Inventory Management System**

This academic project transforms retail transaction history into sales analytics, product-level demand forecasts, customer RFM segments, and explainable reorder suggestions. Inventory quantities are entered separately by the user. The UCI Online Retail source is described in the documentation; upload a local CSV/XLSX copy to run the application.

See `README.md`, `docs/PROJECT_REPORT.md`, `docs/METHODOLOGY.md`, and `docs/ARCHITECTURE.md` for installation, methodology, limitations, and viva notes.""")

{"Dashboard Overview":overview,"Sales Analytics":sales_analytics,"Sales Forecasting":forecasting,"Inventory Management":inventory_page,"Customer Segmentation":customer_page,"Machine Learning Models":models_page,"Model Evaluation":evaluation_page,"Dataset Management":dataset_page,"About Project":about}[page]()
