# A2AI (Advanced Financial Analysis AI) - Production Dockerfile
# 企業生存分析・因果推論・機械学習統合環境

# =============================================================================
# Base Image - Python 3.11 with Ubuntu 22.04
# =============================================================================
FROM python:3.11-slim

# メタデータ
LABEL maintainer="A2AI Development Team"
LABEL version="1.0.0"
LABEL description="Advanced Financial Analysis AI with Survival Analysis and Causal Inference"

# =============================================================================
# 環境変数設定
# =============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

# A2AI固有の環境変数
ENV A2AI_HOME=/app \
    A2AI_DATA=/app/data \
    A2AI_RESULTS=/app/results \
    A2AI_LOGS=/app/logs \
    A2AI_CONFIG=/app/config

# 並列処理・メモリ設定
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4

# =============================================================================
# システムパッケージのインストール
# =============================================================================
RUN apt-get update && apt-get install -y \
    # 基本開発ツール
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # 数値計算・統計ライブラリ依存
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    # グラフィック・可視化依存
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libxft-dev \
    # データベース接続
    libpq-dev \
    libsqlite3-dev \
    # XML/HTMLパース用
    libxml2-dev \
    libxslt1-dev \
    # 圧縮・解凍
    zlib1g-dev \
    liblzma-dev \
    # フォント（日本語対応）
    fonts-noto-cjk \
    fonts-liberation \
    # その他ユーティリティ
    htop \
    vim \
    tree \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 作業ディレクトリ設定
# =============================================================================
WORKDIR ${A2AI_HOME}

# 必要ディレクトリの作成
RUN mkdir -p ${A2AI_DATA} ${A2AI_RESULTS} ${A2AI_LOGS} ${A2AI_CONFIG} \
    && mkdir -p /app/data/raw /app/data/processed /app/data/external \
    && mkdir -p /app/results/models /app/results/analysis_results \
    && mkdir -p /app/results/visualizations /app/results/reports

# =============================================================================
# Python基盤ライブラリのインストール
# =============================================================================
# 最新pipとsetuptoolsにアップグレード
RUN pip install --upgrade pip setuptools wheel

# 数値計算・データ処理基盤
RUN pip install \
    numpy==1.24.3 \
    scipy==1.11.1 \
    pandas==2.0.3 \
    polars==0.18.15 \
    dask[complete]==2023.7.1 \
    numba==0.57.1

# =============================================================================
# 生存分析専門ライブラリ
# =============================================================================
RUN pip install \
    # 主要生存分析ライブラリ
    lifelines==0.27.7 \
    scikit-survival==0.21.0 \
    # 統計分析・仮説検定
    statsmodels==0.14.0 \
    pingouin==0.5.3 \
    # 因果推論ライブラリ
    causalml==0.15.0 \
    dowhy==0.10.1 \
    econml==0.14.1 \
    causalinference==0.1.4

# =============================================================================
# 機械学習・深層学習ライブラリ
# =============================================================================
RUN pip install \
    # 基本機械学習
    scikit-learn==1.3.0 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    catboost==1.2.2 \
    # 深層学習
    torch==2.0.1 \
    torchvision==0.15.2 \
    tensorflow==2.13.0 \
    # アンサンブル・最適化
    optuna==3.3.0 \
    hyperopt==0.2.7 \
    # モデル解釈
    shap==0.42.1 \
    lime==0.2.0.1

# =============================================================================
# データ可視化ライブラリ
# =============================================================================
RUN pip install \
    # 基本可視化
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    bokeh==3.2.1 \
    # 統計可視化
    plotnine==0.12.2 \
    altair==5.0.1 \
    # インタラクティブ
    ipywidgets==8.1.0 \
    voila==0.5.0

# =============================================================================
# データ収集・処理ライブラリ
# =============================================================================
RUN pip install \
    # Web スクレイピング
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    selenium==4.11.2 \
    # API クライアント
    yfinance==0.2.18 \
    alpha-vantage==2.3.1 \
    # データファイル処理
    openpyxl==3.1.2 \
    xlrd==2.0.1 \
    PyPDF2==3.0.1 \
    # データベース
    SQLAlchemy==2.0.19 \
    psycopg2-binary==2.9.7 \
    pymongo==4.4.1

# =============================================================================
# Webアプリケーション・API
# =============================================================================
RUN pip install \
    # Web フレームワーク
    fastapi==0.101.1 \
    uvicorn[standard]==0.23.2 \
    # ダッシュボード
    streamlit==1.25.0 \
    dash==2.13.0 \
    # Jupyter 環境
    jupyter==1.0.0 \
    jupyterlab==4.0.5 \
    notebook==7.0.2

# =============================================================================
# 時系列分析・予測ライブラリ
# =============================================================================
RUN pip install \
    # 時系列分析
    statsforecast==1.5.0 \
    sktime==0.21.1 \
    tsfresh==0.20.1 \
    # Prophet（Facebook時系列）
    prophet==1.1.4 \
    # ARIMA系
    pmdarima==2.0.3

# =============================================================================
# 日本語・テキスト処理
# =============================================================================
RUN pip install \
    # 日本語形態素解析
    janome==0.5.0 \
    mecab-python3==1.0.6 \
    # テキスト処理
    nltk==3.8.1 \
    spacy==3.6.1 \
    # 文書処理
    python-docx==0.8.11 \
    pdfplumber==0.9.0

# =============================================================================
# A2AI専用カスタム要件
# =============================================================================
# A2AI特化ライブラリのインストール
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt && rm /tmp/requirements-docker.txt

# =============================================================================
# 設定ファイルとスクリプトのコピー
# =============================================================================
# 設定ファイル
COPY config/ ${A2AI_CONFIG}/
COPY docker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# =============================================================================
# Jupyter設定
# =============================================================================
# Jupyter設定ディレクトリ作成
RUN mkdir -p /root/.jupyter

# Jupyter設定ファイル
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# =============================================================================
# R言語サポート（統計分析拡張）
# =============================================================================
# R言語のインストール（生存分析の豊富なパッケージ利用）
RUN apt-get update && apt-get install -y \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# 重要なRパッケージのインストール
RUN R -e "install.packages(c('survival', 'survminer', 'rpy2'), repos='https://cloud.r-project.org/')"

# Python-R連携
RUN pip install rpy2==3.5.13

# =============================================================================
# システム最適化とクリーンアップ
# =============================================================================
# 不要ファイルの削除
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /root/.cache

# ログディレクトリの権限設定
RUN chmod -R 755 ${A2AI_LOGS}

# =============================================================================
# ヘルスチェック
# =============================================================================
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import pandas, numpy, lifelines, sklearn; print('A2AI Health Check: OK')" || exit 1

# =============================================================================
# ポート公開
# =============================================================================
# Jupyter Lab
EXPOSE 8888
# FastAPI
EXPOSE 8000
# Streamlit Dashboard
EXPOSE 8501

# =============================================================================
# ボリューム設定
# =============================================================================
VOLUME ["${A2AI_DATA}", "${A2AI_RESULTS}", "${A2AI_LOGS}"]

# =============================================================================
# ユーザー設定
# =============================================================================
# 非rootユーザーの作成（セキュリティ強化）
RUN groupadd -r a2ai && useradd -r -g a2ai -s /bin/bash a2ai
RUN chown -R a2ai:a2ai ${A2AI_HOME}

# =============================================================================
# エントリーポイント
# =============================================================================
USER a2ai
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["jupyter-lab"]

# =============================================================================
# ビルド情報
# =============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.label-schema.build-date=$BUILD_DATE \
        org.label-schema.name="A2AI" \
        org.label-schema.description="Advanced Financial Analysis AI with Survival Analysis" \
        org.label-schema.url="https://github.com/ymk-scsa/A2AI" \
        org.label-schema.vcs-ref=$VCS_REF \
        org.label-schema.vcs-url="https://github.com/ymk-scsa/A2AI" \
        org.label-schema.vendor="A2AI Team" \
        org.label-schema.version=$VERSION \
        org.label-schema.schema-version="1.0"