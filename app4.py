import os
import re
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import traceback
import snowflake.connector
from py2neo import Graph
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from dotenv import load_dotenv
load_dotenv()

# RAG Configuration class
class RAGConfig:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    VECTOR_DB_PATH = "./chroma_db"
    TOP_K_RETRIEVAL = 5


try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError as e:
    CHROMADB_AVAILABLE = False
    st.warning(f"ChromaDB not available: {e}")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.warning(f"LangChain not available: {e}")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        groq_client = None
        GROQ_AVAILABLE = False
        st.warning("GROQ_API_KEY not found in environment variables")
except Exception as e:
    GROQ_AVAILABLE = False
    groq_client = None
    st.warning(f"Groq not available: {e}")

try:
    import networkx as nx
    from py2neo import Graph
    NETWORKX_AVAILABLE = True
    NEO4J_AVAILABLE = True
except Exception as e:
    NETWORKX_AVAILABLE = False
    NEO4J_AVAILABLE = False
    st.warning(f"NetworkX/Neo4j not available: {e}")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError as e:
    PROPHET_AVAILABLE = False
    st.warning(f"Prophet not available: {e}")
#Supply Chain Analytics
class EnhancedIntegratedRAG:
    def __init__(self, conn, graph):
        self.conn = conn
        self.graph = graph
        self.embedding_model = None
        self.vector_store = None
        self.qa_chain = None
        self.documents_processed = False
        self.embeddings = None
        
    def initialize_embeddings(self):
        """Initializing the embedding model"""
        if not LANGCHAIN_AVAILABLE:
            return None
        try:
            if self.embeddings is None:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=RAGConfig.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            return self.embeddings
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            return None

    def extract_product_documents(self):
        """Extracting product information as documents for RAG"""
        if not self.conn:
            return []
        
        try:
            cs = self.conn.cursor()
            query = """
            SELECT 
                p.PRODUCT_ID,
                p.PRODUCT_NAME,
                p.CATEGORY,
                p.PRICE_PER_UNIT,
                p.BRAND,
                p.PRODUCT_DESCRIPTION,
                COUNT(ph.PURCHASE_ID) as purchase_count,
                AVG(ph.QUANTITY) as avg_quantity,
                SUM(ph.QUANTITY * p.PRICE_PER_UNIT) as total_revenue
            FROM PRODUCTS p
            LEFT JOIN PURCHASE_HISTORY ph ON p.PRODUCT_ID = ph.PRODUCT_ID
            GROUP BY p.PRODUCT_ID, p.PRODUCT_NAME, p.CATEGORY, p.PRICE_PER_UNIT, p.BRAND, p.PRODUCT_DESCRIPTION
            """
            cs.execute(query)
            df = pd.DataFrame(cs.fetchall(), columns=[d[0] for d in cs.description])
            
            documents = []
            for _, row in df.iterrows():
                content = f"""
                Product: {row['PRODUCT_NAME']}
                Category: {row['CATEGORY']}
                Brand: {row['BRAND'] if row['BRAND'] else 'Unknown Brand'}
                Price per Unit: ${row['PRICE_PER_UNIT']}
                Description: {row['PRODUCT_DESCRIPTION'] if row['PRODUCT_DESCRIPTION'] else 'No description available'}
                Purchase Statistics:
                - Total purchases: {row['PURCHASE_COUNT'] if row['PURCHASE_COUNT'] else 0}
                - Average quantity per purchase: {row['AVG_QUANTITY'] if row['AVG_QUANTITY'] else 0:.2f}
                - Total revenue generated: ${row['TOTAL_REVENUE'] if row['TOTAL_REVENUE'] else 0:.2f}
                """
                
                metadata = {
                    'product_id': str(row['PRODUCT_ID']),
                    'product_name': row['PRODUCT_NAME'],
                    'category': row['CATEGORY'],
                    'price': float(row['PRICE_PER_UNIT']) if row['PRICE_PER_UNIT'] else 0.0,
                    'brand': row['BRAND'] if row['BRAND'] else 'Unknown',
                    'source': 'product_database'
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
        except Exception as e:
            st.error(f"Error extracting product documents: {e}")
            return []

    def build_vector_store(self):
        """Building the vector store with product documents"""
        if not LANGCHAIN_AVAILABLE:
            st.error("LangChain not available for vector store")
            return False
            
        try:
            embeddings = self.initialize_embeddings()
            if embeddings is None:
                return False
            
            documents = self.extract_product_documents()
            if not documents:
                st.warning("No product documents found")
                return False
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAGConfig.CHUNK_SIZE,
                chunk_overlap=RAGConfig.CHUNK_OVERLAP
            )
            split_docs = text_splitter.split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=RAGConfig.VECTOR_DB_PATH
            )
            
            if groq_client and LANGCHAIN_AVAILABLE:
                llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name="llama3-70b-8192",
                    temperature=0.2
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={"k": RAGConfig.TOP_K_RETRIEVAL}
                    ),
                    return_source_documents=True
                )
            
            self.documents_processed = True
            return True
            
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return False

    def get_customer_purchases(self, customer_id):
        """Getting detailed purchase history for a customer"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            cs = self.conn.cursor()
            query = """
            SELECT 
                cp.CUSTOMER_ID,
                cp.FIRST_NAME,
                cp.LAST_NAME,
                p.PRODUCT_NAME,
                p.CATEGORY,
                p.BRAND,
                ph.QUANTITY,
                ph.TOTAL_AMOUNT,
                ph.PURCHASE_DATE,
                p.PRICE_PER_UNIT
            FROM CUSTOMER_PROFILE cp
            JOIN PURCHASE_HISTORY ph ON cp.CUSTOMER_ID = ph.CUSTOMER_ID
            JOIN PRODUCTS p ON ph.PRODUCT_ID = p.PRODUCT_ID
            WHERE cp.CUSTOMER_ID = %s
            ORDER BY ph.PURCHASE_DATE DESC
            """
            cs.execute(query, (customer_id,))
            result = cs.fetchall()
            
            if result:
                columns = ['Customer_ID', 'First_Name', 'Last_Name', 'Product_Name', 
                          'Category', 'Brand', 'Quantity', 'Total_Amount', 'Purchase_Date', 'Price_Per_Unit']
                return pd.DataFrame(result, columns=columns)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error getting customer purchases: {e}")
            return pd.DataFrame()

    def create_customer_network_graph(self, customer_id):
        """Creating network graph showing customer and their purchases"""
        if not NETWORKX_AVAILABLE:
            return None, None
        
        df = self.get_customer_purchases(customer_id)
        if df.empty:
            return None, None
        
        try:
            G = nx.Graph()
            
            # Adding customer info
            customer_name = f"{df.iloc[0]['First_Name']} {df.iloc[0]['Last_Name']}"
            G.add_node(f"Customer {customer_id}", 
                      node_type='customer', 
                      name=customer_name,
                      size=30)
            
            # Adding product info
            for _, row in df.iterrows():
                product_node = row['Product_Name']
                G.add_node(product_node, 
                          node_type='product', 
                          category=row['Category'],
                          brand=row['Brand'],
                          size=20)
                
                # Adding purchase info
                G.add_edge(f"Customer {customer_id}", product_node,
                          quantity=row['Quantity'],
                          amount=row['Total_Amount'],
                          date=row['Purchase_Date'])
            
            return G, df
            
        except Exception as e:
            st.error(f"Error creating network graph: {e}")
            return None, None

    def visualize_network_graph(self, G, customer_id):
        """Visualizing network graph using plotly"""
        if G is None:
            return None
        
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Preparing node traces
            node_x = []
            node_y = []
            node_text = []
            node_color = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                node_info = G.nodes[node]
                if node_info.get('node_type') == 'customer':
                    node_text.append(f"Customer {customer_id}<br>{node_info.get('name', '')}")
                    node_color.append('red')
                    node_size.append(30)
                else:
                    node_text.append(f"{node}<br>Category: {node_info.get('category', 'N/A')}")
                    node_color.append('lightblue')
                    node_size.append(20)
            
            # Preparing edge traces
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                edge_data = G.edges[edge]
                edge_info.append(f"Quantity: {edge_data.get('quantity', 0)}<br>Amount: ${edge_data.get('amount', 0)}")
            
            
            fig = go.Figure()
            
            # Adding edges
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=2, color='gray'),
                                   hoverinfo='none',
                                   mode='lines'))
            
            # Adding nodes
            fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                   mode='markers+text',
                                   marker=dict(size=node_size,
                                             color=node_color,
                                             line=dict(width=2, color='black')),
                                   text=node_text,
                                   textposition="middle center",
                                   hoverinfo='text',
                                   textfont=dict(size=10)))
            
            fig.update_layout(
                title=f"Customer {customer_id} Purchase Network",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Red node: Customer, Blue nodes: Products purchased",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error visualizing network: {e}")
            return None

    def forecast_sales(self, product_id, periods=30):
        """Forecasting sales for a product using Prophet"""
        if not PROPHET_AVAILABLE or not self.conn:
            return pd.DataFrame()
        
        try:
            cs = self.conn.cursor()
            query = """
            SELECT PURCHASE_DATE, SUM(QUANTITY * PRICE_PER_UNIT) AS sales
            FROM PURCHASE_HISTORY ph
            JOIN PRODUCTS p ON p.PRODUCT_ID = ph.PRODUCT_ID
            WHERE p.PRODUCT_ID = %s
            GROUP BY PURCHASE_DATE
            ORDER BY PURCHASE_DATE
            """
            cs.execute(query, (product_id,))
            result = cs.fetchall()
            
            if not result:
                return pd.DataFrame()
            
            df = pd.DataFrame(result, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'])
            
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            st.error(f"Error in sales forecasting: {e}")
            return pd.DataFrame()

    def predict_churn_for_product(self, product_name, inactivity_days=90):
        """Predicting churn for customers who bought a specific product"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            cs = self.conn.cursor()
            
            # Matching customers with product
            query = """
            SELECT DISTINCT ph.CUSTOMER_ID, MAX(ph.PURCHASE_DATE) AS last_purchase_date
            FROM PURCHASE_HISTORY ph
            JOIN PRODUCTS p ON p.PRODUCT_ID = ph.PRODUCT_ID
            WHERE p.PRODUCT_NAME ILIKE %s
            GROUP BY ph.CUSTOMER_ID
            """
            cs.execute(query, (f'%{product_name}%',))
            result = cs.fetchall()
            
            if not result:
                return pd.DataFrame()
            
            df = pd.DataFrame(result, columns=['CUSTOMER_ID', 'last_purchase_date'])
            df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
            
            reference_date = datetime.today()
            df['days_since_last'] = (reference_date - df['last_purchase_date']).dt.days
            df['churn_flag'] = df['days_since_last'] > inactivity_days
            df['churn_probability'] = (df['days_since_last'] / (inactivity_days * 2)).clip(0, 1)
            
            return df[['CUSTOMER_ID', 'days_since_last', 'churn_probability', 'churn_flag']].sort_values('churn_probability', ascending=False)
            
        except Exception as e:
            st.error(f"Error in churn prediction: {e}")
            return pd.DataFrame()

    def create_forecast_visualization(self, forecast_df, product_name):
        """Creating interactive forecast visualization"""
        if forecast_df.empty:
            return None
        
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sales Forecast', 'Forecast Components'),
                vertical_spacing=0.1
            )
            
            # Forecast plot
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Forecast Confidence interval plot
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(0,100,80,0.2)',
                    fill='tonexty',
                    name='Confidence Interval'
                ),
                row=1, col=1
            )
            
            # Forecast statistics 
            future_data = forecast_df[forecast_df['ds'] > pd.Timestamp.now()]
            if not future_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=['Min Forecast', 'Avg Forecast', 'Max Forecast'],
                        y=[future_data['yhat'].min(), future_data['yhat'].mean(), future_data['yhat'].max()],
                        name='Forecast Stats',
                        marker_color=['red', 'blue', 'green']
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f'Sales Forecast Analysis for {product_name}',
                height=700,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
            fig.update_xaxes(title_text="Forecast Metrics", row=2, col=1)
            fig.update_yaxes(title_text="Value ($)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating forecast visualization: {e}")
            return None

    def create_churn_visualization(self, churn_df, product_name):
        """Creating churn analysis visualization"""
        if churn_df.empty:
            return None
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Churn Probability Distribution', 'Risk Categories', 
                               'Days Since Last Purchase', 'Top At-Risk Customers'),
                specs=[[{"type": "histogram"}, {"type": "pie"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Histogram of churn probabilities
            fig.add_trace(
                go.Histogram(
                    x=churn_df['churn_probability'],
                    nbinsx=20,
                    name='Churn Probability',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # Risk categories pie chart
            high_risk = len(churn_df[churn_df['churn_probability'] > 0.7])
            medium_risk = len(churn_df[(churn_df['churn_probability'] > 0.4) & (churn_df['churn_probability'] <= 0.7)])
            low_risk = len(churn_df[churn_df['churn_probability'] <= 0.4])
            
            fig.add_trace(
                go.Pie(
                    labels=['Low Risk', 'Medium Risk', 'High Risk'],
                    values=[low_risk, medium_risk, high_risk],
                    marker_colors=['green', 'orange', 'red'],
                    name='Risk Distribution'
                ),
                row=1, col=2
            )
            
            # Days since last purchase vs churn probability plot
            fig.add_trace(
                go.Scatter(
                    x=churn_df['days_since_last'],
                    y=churn_df['churn_probability'],
                    mode='markers',
                    marker=dict(
                        color=churn_df['churn_probability'],
                        colorscale='RdYlGn_r',
                        size=8,
                        colorbar=dict(title="Churn Probability")
                    ),
                    name='Customer Risk'
                ),
                row=2, col=1
            )
            
            # Top 10 at-risk customers
            top_risk = churn_df.head(10)
            fig.add_trace(
                go.Bar(
                    x=top_risk['CUSTOMER_ID'].astype(str),
                    y=top_risk['churn_probability'],
                    marker_color='red',
                    name='Top At-Risk'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Churn Risk Analysis for {product_name} Customers',
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating churn visualization: {e}")
            return None

    # PRODUCT MATCHING
    def extract_product_name_flexible(self, question):
        """More flexible product name extraction"""
        
        patterns = [
            r'(?:product|for)\s+(["\']?)([^"\'?\n,]+)\1',  # Original pattern
            r'forecast\s+(?:for\s+)?(["\']?)([^"\'?\n,]+)\1',  # forecast for X
            r'predict\s+sales\s+(?:for\s+)?(["\']?)([^"\'?\n,]+)\1',  # predict sales for X
            r'sales\s+(?:for\s+)?(["\']?)([^"\'?\n,]+)\1',  # sales for X
            r'"([^"]+)"',  # Anything in quotes
            r"'([^']+)'",  # Anything in single quotes
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    return match.group(2).strip()
                else:
                    return match.group(1).strip()
        
        # If no pattern matches, extracting the last meaningful word
        words = question.split()
        for word in reversed(words):
            if len(word) > 2 and word.lower() not in ['for', 'the', 'and', 'product', 'forecast', 'sales']:
                return word
        
        return None

    def find_product_id(self, product_name):
        """Finding product ID with multiple search strategies"""
        if not self.conn:
            return None, None
        
        try:
            cs = self.conn.cursor()
            
            #Exact match (case insensitive)
            cs.execute("SELECT PRODUCT_ID, PRODUCT_NAME FROM PRODUCTS WHERE UPPER(PRODUCT_NAME) = UPPER(%s)", (product_name,))
            result = cs.fetchone()
            if result:
                print(f"DEBUG: Found exact match - ID: {result[0]}, Name: '{result[1]}'")
                return result
            
            #Partial match
            cs.execute("SELECT PRODUCT_ID, PRODUCT_NAME FROM PRODUCTS WHERE PRODUCT_NAME ILIKE %s LIMIT 1", (f'%{product_name}%',))
            result = cs.fetchone()
            if result:
                print(f"DEBUG: Found partial match - ID: {result[0]}, Name: '{result[1]}'")
                return result
            
            #Word-based search
            words = product_name.split()
            for word in words:
                if len(word) > 2:  # Only search for meaningful words
                    cs.execute("SELECT PRODUCT_ID, PRODUCT_NAME FROM PRODUCTS WHERE PRODUCT_NAME ILIKE %s LIMIT 1", (f'%{word}%',))
                    result = cs.fetchone()
                    if result:
                        print(f"DEBUG: Found word match for '{word}' - ID: {result[0]}, Name: '{result[1]}'")
                        return result
            
            print(f"DEBUG: No product found for '{product_name}'")
            return None, None
            
        except Exception as e:
            print(f"DEBUG: Error in find_product_id: {e}")
            return None, None

    def debug_product_search(self, search_term):
        """Debugging function to see what products are available"""
        if not self.conn:
            return []
        
        try:
            cs = self.conn.cursor()
            cs.execute("""
            SELECT PRODUCT_ID, PRODUCT_NAME, CATEGORY 
            FROM PRODUCTS 
            WHERE PRODUCT_NAME ILIKE %s 
            OR PRODUCT_NAME ILIKE %s
            ORDER BY PRODUCT_NAME
            LIMIT 10
            """, (f'%{search_term}%', f'{search_term}%'))
            
            results = cs.fetchall()
            print(f"DEBUG: Products matching '{search_term}':")
            for product_id, name, category in results:
                print(f"  - ID: {product_id}, Name: '{name}', Category: {category}")
            
            return results
        except Exception as e:
            print(f"DEBUG: Error searching products: {e}")
            return []

    def get_available_products_sample(self):
        """Getting a sample of available products for error messages"""
        if not self.conn:
            return []
        
        try:
            cs = self.conn.cursor()
            cs.execute("SELECT PRODUCT_NAME FROM PRODUCTS ORDER BY PRODUCT_NAME LIMIT 10")
            return [row[0] for row in cs.fetchall()]
        except Exception as e:
            print(f"DEBUG: Error getting product samples: {e}")
            return []

    def integrated_query(self, question: str):
        """Main integrated query method that handles all types of analyses"""
        lower_q = question.lower()
        results = {
            "answer": "",
            "visualizations": [],
            "tables": [],
            "entities_found": []
        }
        
        try:
            # Checking for sales forecasting queries
            if any(keyword in lower_q for keyword in ['forecast', 'predict sales', 'sales prediction']):
                
                # Using the flexible extraction method
                product_name = self.extract_product_name_flexible(question)
                
                if product_name and self.conn:
                    print(f"DEBUG: Extracted product name: '{product_name}'")
                    
                    # Displaying the products that match
                    self.debug_product_search(product_name)
                    
                    #Product matching
                    product_result = self.find_product_id(product_name)
                    
                    if product_result and product_result[0]:
                        product_id, actual_product_name = product_result
                        print(f"DEBUG: Using product ID {product_id} for '{actual_product_name}'")
                        
                        forecast_df = self.forecast_sales(product_id, 30)
                        
                        if not forecast_df.empty:
                            
                            forecast_viz = self.create_forecast_visualization(forecast_df, actual_product_name)
                            
                            if forecast_viz:
                                results["visualizations"].append(("forecast_plot", forecast_viz))
                            
                            results["tables"].append(("forecast_data", forecast_df.tail(15)))
                            
                            #Summary
                            future_sales = forecast_df[forecast_df['ds'] > pd.Timestamp.now()]
                            if not future_sales.empty:
                                avg_forecast = future_sales['yhat'].mean()
                                total_forecast = future_sales['yhat'].sum()
                                trend = "increasing" if future_sales['yhat'].iloc[-1] > future_sales['yhat'].iloc[0] else "decreasing"
                                
                                results["answer"] = f"""
**Sales Forecast Analysis for {actual_product_name}**

**Key Metrics:**
- Average daily forecast: ${avg_forecast:.2f}
- Total 30-day forecast: ${total_forecast:.2f}
- Trend: {trend}

**Analysis:**
The sales forecast model predicts the expected performance for {actual_product_name} over the next 30 days. 
The confidence intervals show the range of expected values, helping with inventory planning and revenue projections.
                                """
                            else:
                                results["answer"] = f"Sales forecast generated for {actual_product_name}, but no future data available for analysis."
                        else:
                            results["answer"] = f"Unable to generate forecast for {actual_product_name} - insufficient historical data."
                    else:
                        available_products = self.get_available_products_sample()
                        results["answer"] = f"""
**Product '{product_name}' not found in the database.**

 **Available products include:**
{chr(10).join([f"- {prod}" for prod in available_products[:5]])}
{'...' if len(available_products) > 5 else ''}

 **Tip:** Try using the exact product name or a simpler term.

**Debug Info:** Searched for patterns matching '{product_name}'
                        """
                else:
                    results["answer"] = "Could not extract product name from your query. Please specify the product name more clearly."
            
            # Checking for churn analysis queries
            elif any(keyword in lower_q for keyword in ['churn', 'churn analysis', 'customer retention']):
                # Using the product extraction
                product_name = self.extract_product_name_flexible(question)
                
                if product_name:
                    print(f"DEBUG: Churn analysis for product: '{product_name}'")
                    churn_df = self.predict_churn_for_product(product_name, 90)
                    
                    if not churn_df.empty:
                        # Creating churn visualization
                        churn_viz = self.create_churn_visualization(churn_df, product_name)
                        
                        if churn_viz:
                            results["visualizations"].append(("churn_plot", churn_viz))
                        
                        results["tables"].append(("churn_data", churn_df.head(20)))
                        
                        #Summary
                        high_risk = len(churn_df[churn_df['churn_probability'] > 0.7])
                        medium_risk = len(churn_df[(churn_df['churn_probability'] > 0.4) & (churn_df['churn_probability'] <= 0.7)])
                        low_risk = len(churn_df[churn_df['churn_probability'] <= 0.4])
                        
                        results["answer"] = f"""
**Churn Risk Analysis for {product_name} Customers**

**Risk Distribution:**
- High Risk (>70%): {high_risk} customers ({high_risk/len(churn_df)*100:.1f}%)
- Medium Risk (40-70%): {medium_risk} customers ({medium_risk/len(churn_df)*100:.1f}%)
- Low Risk (<40%): {low_risk} customers ({low_risk/len(churn_df)*100:.1f}%)

 **Key Insights:**
- Total customers analyzed: {len(churn_df)}
- Highest churn probability: {churn_df['churn_probability'].max():.2f}
- Average days since last purchase: {churn_df['days_since_last'].mean():.1f} days

**Recommendations:**
Focus retention efforts on the {high_risk} high-risk customers. Consider targeted promotions or engagement campaigns.
                        """
                    else:
                        results["answer"] = f"No customers found who purchased {product_name}."
                else:
                    results["answer"] = "Could not extract product name for churn analysis. Please specify the product name more clearly."
            
            
            
            # Checking for customer purchase queries
            elif any(keyword in lower_q for keyword in ['customer', 'bought', 'purchased', 'buy']):
                customer_match = re.search(r'customer\s+(\d+)', question, re.IGNORECASE)
                if customer_match:
                    customer_id = int(customer_match.group(1))
                    
                    # Getting customer purchases
                    purchases_df = self.get_customer_purchases(customer_id)
                    
                    if not purchases_df.empty:
                        # Creating network graph
                        G, _ = self.create_customer_network_graph(customer_id)
                        if G:
                            network_viz = self.visualize_network_graph(G, customer_id)
                            if network_viz:
                                results["visualizations"].append(("network_graph", network_viz))
                        
                        results["tables"].append(("customer_purchases", purchases_df))
                        
                        #Summary
                        total_spent = purchases_df['Total_Amount'].sum()
                        unique_products = purchases_df['Product_Name'].nunique()
                        favorite_category = purchases_df['Category'].mode().iloc[0] if not purchases_df['Category'].mode().empty else "N/A"
                        
                        results["answer"] = f"""
**Purchase Analysis for Customer {customer_id}**

 **Customer Info:**
- Name: {purchases_df.iloc[0]['First_Name']} {purchases_df.iloc[0]['Last_Name']}
- Total Spent: ${total_spent:.2f}
- Unique Products Purchased: {unique_products}
- Favorite Category: {favorite_category}

 **Purchase Summary:**
- Total Purchases: {len(purchases_df)}
- Average Order Value: ${purchases_df['Total_Amount'].mean():.2f}
- Most Recent Purchase: {purchases_df['Purchase_Date'].max()}

**The network graph shows the relationship between the customer and their purchased products, with connections representing individual purchases.**
                        """
                    else:
                        results["answer"] = f"No purchase history found for Customer {customer_id}."
            
            # RAG Query
            else:
                # Getting vector and graph context
                vector_result = self.get_vector_context(question)
                graph_result = self.get_graph_context(question)
                
                # Combining contexts
                contexts = []
                if "context" in vector_result and not vector_result.get("error"):
                    contexts.append(f"VECTOR DATABASE CONTEXT:\n{vector_result['context']}")
                
                if graph_result.get("context"):
                    contexts.append(f"KNOWLEDGE GRAPH CONTEXT:\n{graph_result['context']}")
                
                combined_context = "\n\n" + "="*50 + "\n\n".join(contexts) if contexts else "No relevant context found"
                
                # Generating comprehensive answer using LLM
                if groq_client and contexts:
                    prompt = f"""
                    You are an intelligent retail analytics assistant with access to comprehensive product and customer data.
                    
                    Based on the following context from multiple data sources, provide a detailed and insightful answer:
                    
                    {combined_context}
                    
                    Question: {question}
                    
                    Instructions:
                    - Synthesize information from both vector database and knowledge graph
                    - Provide specific data points where available
                    - Highlight relationships and patterns
                    - Be comprehensive but concise
                    - If data is limited, clearly state what information is available
                    """
                    
                    response = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are an expert retail analyst with access to comprehensive product and customer relationship data."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                        temperature=0.2
                    )
                    
                    results["answer"] = response.choices[0].message.content
                    results["entities_found"] = graph_result.get("entities", [])
                else:
                    # Fallback without LLM
                    results["answer"] = f"Based on available data:\n\n{combined_context}"
                    results["entities_found"] = graph_result.get("entities", [])
                
        except Exception as e:
            results["answer"] = f"Error processing query: {e}"
            import traceback
            print(f"ERROR: {traceback.format_exc()}")
        
        return results

    def get_vector_context(self, question: str):
        """Getting context from vector database"""
        if not self.documents_processed:
            if not self.build_vector_store():
                return {"error": "Failed to build vector store"}
        
        try:
            if self.vector_store:
                docs = self.vector_store.similarity_search(question, k=RAGConfig.TOP_K_RETRIEVAL)
                return {
                    "context": "\n\n".join([doc.page_content for doc in docs]),
                    "source_documents": docs
                }
            else:
                return {"error": "Vector store not available"}
        except Exception as e:
            return {"error": f"Vector search failed: {e}"}

    def get_graph_context(self, question: str):
        """Getting context from knowledge graph"""
        if not NEO4J_AVAILABLE or not self.graph:
            return {"context": "Knowledge graph not available", "entities": []}
        
        try:
            # Extracting entities from question
            entities = self._extract_entities(question)
            
            # Building context from knowledge graph
            graph_contexts = []
            for entity, entity_type in entities:
                context = self._build_graph_context(entity, entity_type)
                if context:
                    graph_contexts.append(context)
            
            combined_context = "\n\n".join(graph_contexts) if graph_contexts else "No relevant graph connections found"
            
            return {
                "context": combined_context,
                "entities": entities
            }
            
        except Exception as e:
            return {"context": f"Graph query error: {e}", "entities": []}

    def _build_graph_context(self, entity_name, entity_type):
        """Building context from graph for specific entity"""
        try:
            if entity_type == "product":
                query = """
                MATCH (p:Product {product_name: $name})
                OPTIONAL MATCH (p)<-[r:PURCHASED]-(c:Customer)
                OPTIONAL MATCH (p)-[:IN_CATEGORY]->(cat:Category)
                RETURN p, collect(DISTINCT c.customer_id)[..5] as customer_ids, 
                       collect(DISTINCT cat.name) as categories,
                       count(DISTINCT r) as purchase_count
                """
                result = self.graph.run(query, name=entity_name).data()
                
                if result and result[0]['p']:
                    product = result[0]['p']
                    customers = result[0]['customer_ids']
                    categories = result[0]['categories']
                    purchases = result[0]['purchase_count']
                    
                    context = f"Graph Analysis for {entity_name}:\n"
                    context += f"- Purchased by {purchases} customers\n"
                    context += f"- Categories: {', '.join(categories) if categories else 'None'}\n"
                    context += f"- Top customers: {', '.join(map(str, customers)) if customers else 'None'}\n"
                    
                    return context
                    
            elif entity_type == "customer":
                query = """
                MATCH (c:Customer)
                WHERE toString(c.customer_id) = $name
                OPTIONAL MATCH (c)-[r:PURCHASED]->(p:Product)
                RETURN c, collect(DISTINCT p.product_name)[..5] as products,
                       count(DISTINCT r) as purchase_count
                """
                result = self.graph.run(query, name=str(entity_name)).data()
                
                if result and result[0]['c']:
                    customer = result[0]['c']
                    products = result[0]['products']
                    purchases = result[0]['purchase_count']
                    
                    context = f"Graph Analysis for Customer {entity_name}:\n"
                    context += f"- Total purchases: {purchases}\n"
                    context += f"- Products: {', '.join(products) if products else 'None'}\n"
                    
                    return context
                    
        except Exception as e:
            return f"Graph context error for {entity_name}: {e}"
        
        return None

    def _extract_entities(self, text):
        """Extracting entities from text"""
        entities = []
        
        # Extracting product names 
        if self.conn:
            try:
                cs = self.conn.cursor()
                cs.execute("SELECT PRODUCT_NAME FROM PRODUCTS")
                products = [row[0] for row in cs.fetchall()]
                
                for product in products:
                    if product and re.search(rf'\b{re.escape(str(product))}\b', text, re.IGNORECASE):
                        entities.append((product, "product"))
            except:
                pass
        
        # Extracting customer IDs
        customer_matches = re.findall(r'customer(?:\s+id)?\s*[:#]?\s*(\d+)', text, re.IGNORECASE)
        for match in customer_matches:
            entities.append((match, "customer"))
        
        return entities


# Streamlit UI Integration
def create_enhanced_ui():
    """Creating the enhanced UI with integrated analytics"""
    
    st.set_page_config(
        page_title="Enhanced Hybrid RAG Retail Assistant",
        page_icon="",
        layout="wide"
    )
    
    st.title("Enhanced Hybrid RAG Retail Assistant")
    st.markdown("**Unified Intelligence**: Combining Vector Embeddings + Knowledge Graph Relationships + Advanced Analytics")
    
    # Initializing connections
    conn = create_snowflake_connection()
    graph = create_neo4j_graph()   
    # Initializing the enhanced RAG system
    @st.cache_resource
    def initialize_enhanced_rag():
        return EnhancedIntegratedRAG(conn, graph)
    
    enhanced_rag = initialize_enhanced_rag()
    
    # Sidebar status
    st.sidebar.header("System Status")
    st.sidebar.write(f"Snowflake: {'' if conn else ''} {'Connected' if conn else 'Not connected'}")
    st.sidebar.write(f"Neo4j: {'' if NEO4J_AVAILABLE and graph else ''} {'Connected' if NEO4J_AVAILABLE and graph else 'Not available'}")
    st.sidebar.write(f"Groq LLM: {'' if groq_client else ''} {'Available' if groq_client else 'Not configured'}")
    st.sidebar.write(f"Embeddings: {'' if LANGCHAIN_AVAILABLE else ''} {'Available' if LANGCHAIN_AVAILABLE else 'Install requirements'}")
    
    # Main Interface
    st.header(" Intelligent Retail Query with Integrated Analytics")
    st.markdown("Ask comprehensive questions and get visualizations, tables, and analysis automatically!")
    
    # example queries
    with st.expander(" Example Queries with Integrated Analytics"):
        st.markdown("""
        **Sales Forecasting:**
        - "Show me sales forecast for product Bread"
        - "Predict sales for Laptop"
        - "Forecast revenue for Apple products"
        
        **Churn Analysis:**
        - "Show me churn analysis for product Apple"
        - "Analyze customer retention for Bread customers"
        - "Which customers might stop buying Laptop?"
        
        **Customer Purchase Networks:**
        - "What did customer 5 buy?" (Shows network graph)
        - "Show purchase history for customer 10 with visualization"
        - "Analyze customer 3 purchases with network"
        
        **General Analytics:**
        - "Tell me about Bread product comprehensively"
        - "Compare electronics vs food categories"
        - "Show relationships between high-value customers and premium products"
        """)
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="e.g., 'Show me sales forecast for product Bread' or 'What did customer 5 buy?'"
    )
    
    # Query execution
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("Analyze with Enhanced RAG", type="primary"):
            if not user_query.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Processing with integrated analytics..."):
                    results = enhanced_rag.integrated_query(user_query)
                
                # Displaying main answer
                st.subheader("Analysis Results")
                st.write(results["answer"])
                
                # Displaying visualizations
                if results.get("visualizations"):
                    st.subheader("Interactive Visualizations")
                    for viz_name, viz_fig in results["visualizations"]:
                        if viz_fig:
                            st.plotly_chart(viz_fig, use_container_width=True)
                
                # Displaying tables
                if results.get("tables"):
                    st.subheader("Data Tables")
                    for table_name, table_df in results["tables"]:
                        st.write(f"**{table_name.replace('_', ' ').title()}:**")
                        st.dataframe(table_df, use_container_width=True)
                
                # Displaying entities found
                if results.get("entities_found"):
                    with st.expander("Entities Detected"):
                        for entity, entity_type in results["entities_found"]:
                            st.write(f"- **{entity}** _{entity_type}_")
    
    with col2:
        if st.button("Initialize System"):
            with st.spinner("Buildinng RAG system..."):
                success = enhanced_rag.build_vector_store()
                if success:
                    st.success("System ready!")
                else:
                    st.error("Initialization failed")
    
    # Feature highlights
    st.markdown("---")
    st.header("Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ** Forecasting**
        - Detects forecast queries
        - Generates Prophet models
        - Interactive visualizations
        - Confidence intervals
        """)
    
    with col2:
        st.markdown("""
        **Churn Analysis**
        - Product-specific churn
        - Risk categorization
        - Visual dashboards
        - Actionable insights
        """)
    
    with col3:
        st.markdown("""
        ** Network Graphs**
        - Customer-product relationships
        - Interactive visualizations
        - Purchase patterns
        - Connection analysis
        """)
    
    with col4:
        st.markdown("""
        **Unified Intelligence**
        - Multi-source synthesis
        - Contextual reasoning
        - Automatic visualization
        - Comprehensive analysis
        """)
    
    return enhanced_rag

# Helper functions 
@st.cache_resource
def create_snowflake_connection():
    """Creating and cache Snowflake connection"""
    try:
        conn = snowflake.connector.connect(
            user='MOMO',
            password='Snowflake2025@#',
            account='obdgxmi-gf38058',
            warehouse='RETAIL_WH',
            database='RETAIL_DB',
            schema='RETAIL_SCHEMA',
            role='ACCOUNTADMIN'
        )
        return conn
    except Exception as e:
        st.error(f"Failed to connect to Snowflake: {e}")
        return None

@st.cache_resource
def create_neo4j_graph():
    """Creating and cache Neo4j graph connection"""
    try:
        return Graph("neo4j://127.0.0.1:7687", auth=("neo4j", "retail2025@#"))
    except Exception as e:
        st.warning(f"Failed to connect to Neo4j: {e}")
        return None

# Run the UI
if __name__ == "__main__":
    create_enhanced_ui()