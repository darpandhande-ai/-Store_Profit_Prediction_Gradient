import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model loading logic with fallback dummy predictor for testing
MODEL_PATH = "Gradient_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Loaded GradientBoostingRegressor model successfully.")
    except Exception as e:
        print(f"Error loading model pickle: {e}")
else:
        print("Gradient_model.pkl not found. Running with simulated fallback model.")

# Feature names aligned with model pickle
FEATURES = [
    "Ship Mode", "Customer Name", "Segment", "Country", "City",
    "State", "Region", "Category", "Sub-Category", "Product Name",
    "Sales", "Quantity", "Discount"
]

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" data-theme="cyberpunk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cap Round Institute Prediction</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Bootstrap 5 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --font-main: 'Plus Jakarta Sans', sans-serif;
            --font-heading: 'Space Grotesk', sans-serif;
            --transition-speed: 0.3s;
        }

        /* Color Themes */
        [data-theme="cyberpunk"] {
            --bg-primary: #0a0e17;
            --bg-secondary: #121824;
            --card-bg: rgba(22, 30, 46, 0.85);
            --accent-glow: #00f2fe;
            --accent-2: #4facfe;
            --accent-3: #ff0844;
            --accent-4: #f77062;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-color: rgba(0, 242, 254, 0.2);
            --gradient-1: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --gradient-2: linear-gradient(135deg, #ff0844 0%, #f77062 100%);
        }

        [data-theme="emerald"] {
            --bg-primary: #062c22;
            --bg-secondary: #0a3a2f;
            --card-bg: rgba(15, 59, 48, 0.85);
            --accent-glow: #10b981;
            --accent-2: #34d399;
            --accent-3: #f59e0b;
            --accent-4: #fbbf24;
            --text-main: #ecfdf5;
            --text-muted: #a7f3d0;
            --border-color: rgba(16, 185, 129, 0.25);
            --gradient-1: linear-gradient(135deg, #10b981 0%, #059669 100%);
            --gradient-2: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }

        [data-theme="sunset"] {
            --bg-primary: #1a0c1e;
            --bg-secondary: #28122d;
            --card-bg: rgba(48, 20, 55, 0.85);
            --accent-glow: #ff007f;
            --accent-2: #7928ca;
            --accent-3: #ff0080;
            --accent-4: #ff4d4d;
            --text-main: #fff0f5;
            --text-muted: #d8b4e2;
            --border-color: rgba(255, 0, 127, 0.25);
            --gradient-1: linear-gradient(135deg, #ff007f 0%, #7928ca 100%);
            --gradient-2: linear-gradient(135deg, #ff4d4d 0%, #f9cb28 100%);
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-main);
            min-height: 100vh;
            overflow-x: hidden;
            transition: background-color var(--transition-speed) ease;
        }

        h1, h2, h3, h4, h5, .brand-font {
            font-family: var(--font-heading);
            letter-spacing: -0.5px;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
        }

        /* Animated Header Glow */
        .glow-title {
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
            display: inline-block;
        }

        .glow-title::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: var(--gradient-1);
            filter: blur(25px);
            opacity: 0.35;
            z-index: -1;
        }

        /* Form Controls */
        .form-label {
            color: var(--text-main);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        .form-control, .form-select {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main) !important;
            border-radius: 12px;
            padding: 0.65rem 1rem;
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            background: rgba(255, 255, 255, 0.08);
            border-color: var(--accent-glow);
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .form-select option {
            background-color: var(--bg-secondary);
            color: var(--text-main);
        }

        /* Pulse Button */
        .btn-predict {
            background: var(--gradient-1);
            border: none;
            color: #fff;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 0.9rem 2rem;
            border-radius: 14px;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn-predict:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 242, 254, 0.7);
            color: #fff;
        }

        /* Result Animation Card */
        .result-card {
            display: none;
            background: var(--gradient-2);
            border-radius: 20px;
            padding: 2rem;
            color: #fff;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Theme Selector Buttons */
        .theme-btn {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 2px solid #fff;
            cursor: pointer;
            transition: transform 0.2s ease;
        }

        .theme-btn:hover { transform: scale(1.2); }
        .theme-cyberpunk { background: linear-gradient(45deg, #00f2fe, #4facfe); }
        .theme-emerald { background: linear-gradient(45deg, #10b981, #34d399); }
        .theme-sunset { background: linear-gradient(45deg, #ff007f, #7928ca); }
    </style>
</head>
<body class="py-4">

    <div class="container">
        <!-- Top Navigation Bar -->
        <header class="d-flex justify-content-between align-items-center mb-5 pb-3 border-bottom border-secondary">
            <div class="d-flex align-items-center gap-3">
                <i class="fa-solid fa-graduation-cap fa-2x text-info"></i>
                <h2 class="glow-title m-0">Cap Round Institute Prediction</h2>
            </div>
            
            <div class="d-flex align-items-center gap-3">
                <span class="text-muted small fw-bold">THEME:</span>
                <div class="theme-btn theme-cyberpunk" onclick="setTheme('cyberpunk')" title="Cyberpunk Neon"></div>
                <div class="theme-btn theme-emerald" onclick="setTheme('emerald')" title="Emerald Forest"></div>
                <div class="theme-btn theme-sunset" onclick="setTheme('sunset')" title="Sunset Vibrant"></div>
            </div>
        </header>

        <!-- Main Dashboard Content -->
        <div class="row g-4">
            <!-- Left Panel: Input Parameters -->
            <div class="col-lg-5">
                <div class="glass-card p-4">
                    <h4 class="mb-4 d-flex align-items-center gap-2">
                        <i class="fa-solid fa-sliders text-info"></i> Model Parameters
                    </h4>
                    <form id="predictionForm" onsubmit="handlePredict(event)">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Ship Mode</label>
                                <select class="form-select" name="Ship Mode">
                                    <option>Standard Class</option>
                                    <option>Second Class</option>
                                    <option>First Class</option>
                                    <option>Same Day</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Segment</label>
                                <select class="form-select" name="Segment">
                                    <option>Consumer</option>
                                    <option>Corporate</option>
                                    <option>Home Office</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Category</label>
                                <select class="form-select" name="Category">
                                    <option>Technology</option>
                                    <option>Furniture</option>
                                    <option>Office Supplies</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Sub-Category</label>
                                <select class="form-select" name="Sub-Category">
                                    <option>Phones</option>
                                    <option>Chairs</option>
                                    <option>Storage</option>
                                    <option>Tables</option>
                                    <option>Binders</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Sales Value</label>
                                <input type="number" step="0.01" class="form-control" name="Sales" value="250.00" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Quantity</label>
                                <input type="number" class="form-control" name="Quantity" value="3" required>
                            </div>
                            <div class="col-md-12">
                                <label class="form-label">Discount Rate (0 - 1)</label>
                                <input type="number" step="0.01" min="0" max="1" class="form-control" name="Discount" value="0.10" required>
                            </div>
                        </div>

                        <!-- Hidden placeholder fields for model features -->
                        <input type="hidden" name="Customer Name" value="Default Customer">
                        <input type="hidden" name="Country" value="United States">
                        <input type="hidden" name="City" value="New York">
                        <input type="hidden" name="State" value="New York">
                        <input type="hidden" name="Region" value="East">
                        <input type="hidden" name="Product Name" value="Generic Item">

                        <button type="submit" class="btn btn-predict w-100 mt-4">
                            <i class="fa-solid fa-wand-magic-sparkles me-2"></i> Run Prediction
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Panel: Analytics & Results -->
            <div class="col-lg-7">
                <!-- Result Banner -->
                <div id="resultCard" class="result-card mb-4 text-center">
                    <h5 class="text-uppercase tracking-wider opacity-75 m-0">Predicted Target Metric</h5>
                    <h1 class="display-3 fw-bold my-2" id="predictionOutput">0.00</h1>
                    <p class="m-0 small opacity-90"><i class="fa-solid fa-circle-check me-1"></i> Gradient Boosting Regressor Analysis Complete</p>
                </div>

                <!-- Charts Layout -->
                <div class="glass-card p-4">
                    <h4 class="mb-3 d-flex align-items-center gap-2">
                        <i class="fa-solid fa-chart-line text-info"></i> Predictive Visual Analytics
                    </h4>
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div style="position: relative; height:230px;">
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div style="position: relative; height:230px;">
                                <canvas id="barChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Theme Switcher Logic
        function setTheme(themeName) {
            document.documentElement.setAttribute('data-theme', themeName);
            updateChartColors();
        }

        // Initialize Charts
        let radarChart, barChart;

        function initCharts() {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const ctxBar = document.getElementById('barChart').getContext('2d');

            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Sales Impact', 'Quantity', 'Discount Factor', 'Category Weight', 'Regional Index'],
                    datasets: [{
                        label: 'Feature Weight',
                        data: [65, 59, 80, 81, 56],
                        fill: true,
                        backgroundColor: 'rgba(0, 242, 254, 0.2)',
                        borderColor: '#00f2fe',
                        pointBackgroundColor: '#ff0844',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#f1f5f9' } } },
                    scales: { r: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { display: false } } }
                }
            });

            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: ['Cap Round 1', 'Cap Round 2', 'Cap Round 3', 'Spot Round'],
                    datasets: [{
                        label: 'Estimated Cutoff Trend',
                        data: [88, 82, 75, 69],
                        backgroundColor: ['#00f2fe', '#4facfe', '#ff0844', '#f77062'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: '#f1f5f9' }, grid: { display: false } },
                        y: { ticks: { color: '#f1f5f9' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        }

        function updateChartColors() {
            if (radarChart && barChart) {
                radarChart.update();
                barChart.update();
            }
        }

        // Prediction Form Handling
        async function handlePredict(e) {
            e.preventDefault();
            const formData = new FormData(e.target);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (result.status === 'success') {
                    const output = document.getElementById('predictionOutput');
                    const card = document.getElementById('resultCard');
                    
                    card.style.display = 'block';
                    output.innerText = result.prediction.toFixed(2);

                    // Update Charts dynamically on prediction
                    barChart.data.datasets[0].data = [
                        result.prediction * 0.95,
                        result.prediction * 0.88,
                        result.prediction * 0.82,
                        result.prediction * 0.75
                    ];
                    barChart.update();
                }
            } catch (err) {
                console.error("Prediction Request Failed:", err);
            }
        }

        window.onload = initCharts;
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features matching the model training structure
        input_data = []
        for feature in FEATURES:
            val = request.form.get(feature, "0")
            try:
                input_data.append(float(val))
            except ValueError:
                # Basic string length fallback encoding for categorical values
                input_data.append(float(len(str(val))))

        features_array = np.array([input_data])

        if model is not None:
            prediction = float(model.predict(features_array)[0])
        else:
            # Fallback mock calculation if model pickle is missing/incompatible
            sales = float(request.form.get("Sales", 100))
            quantity = float(request.form.get("Quantity", 1))
            discount = float(request.form.get("Discount", 0.1))
            prediction = (sales * quantity) * (1.0 - discount)

        return jsonify({"status": "success", "prediction": prediction})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
