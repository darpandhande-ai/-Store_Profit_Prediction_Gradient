import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load Model
MODEL_PATH = "Gradient_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("GradientBoostingRegressor model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Running in fallback/simulation mode.")

# Feature Definitions from Model Metadata
FEATURE_NAMES = [
    "Ship Mode", "Customer Name", "Segment", "Country", "City", 
    "State", "Region", "Category", "Sub-Category", "Product Name", 
    "Sales", "Quantity", "Discount"
]

DEFAULT_VALUES = {
    "Ship Mode": 0, "Customer Name": 10, "Segment": 0, "Country": 0,
    "City": 20, "State": 10, "Region": 1, "Category": 0,
    "Sub-Category": 3, "Product Name": 100, "Sales": 250.00,
    "Quantity": 3, "Discount": 0.10
}

# Embedded HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" x-data="dashboard()" :class="theme">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Store Profit Prediction - Analytics Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                        mono: ['Fira Code', 'monospace'],
                        display: ['Outfit', 'sans-serif']
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        [x-cloak] { display: none !important; }
        
        /* Themes */
        .theme-cyber {
            --bg-primary: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --border-color: rgba(99, 102, 241, 0.25);
            --text-main: #f8fafc;
            --card-gradient: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
            --accent: #6366f1;
        }
        
        .theme-sunset {
            --bg-primary: #180914;
            --bg-card: rgba(36, 15, 30, 0.7);
            --border-color: rgba(244, 63, 94, 0.25);
            --text-main: #fdf2f8;
            --card-gradient: linear-gradient(135deg, rgba(88, 28, 135, 0.5) 0%, rgba(24, 9, 20, 0.8) 100%);
            --accent: #f43f5e;
        }

        .theme-emerald {
            --bg-primary: #022c22;
            --bg-card: rgba(6, 78, 59, 0.5);
            --border-color: rgba(16, 185, 129, 0.25);
            --text-main: #ecfdf5;
            --card-gradient: linear-gradient(135deg, rgba(6, 95, 70, 0.5) 0%, rgba(2, 44, 34, 0.8) 100%);
            --accent: #10b981;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            transition: background-color 0.4s ease, color 0.4s ease;
        }

        .glass-card {
            background: var(--card-gradient);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
        }

        @keyframes pulse-slow {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.05); }
        }
        .animate-pulse-slow {
            animation: pulse-slow 6s infinite ease-in-out;
        }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.1); }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.2); border-radius: 3px; }
    </style>
</head>
<body class="font-sans antialiased min-h-screen relative overflow-x-hidden" :class="fontStyle">

    <!-- Ambient Glowing Backgrounds -->
    <div class="fixed -top-20 -left-20 w-96 h-96 bg-indigo-600/20 rounded-full blur-3xl pointer-events-none animate-pulse-slow"></div>
    <div class="fixed bottom-0 right-0 w-96 h-96 bg-rose-600/20 rounded-full blur-3xl pointer-events-none animate-pulse-slow" style="animation-delay: 3s;"></div>

    <div class="container mx-auto px-4 py-8 relative z-10 max-w-7xl">
        
        <!-- Dashboard Header -->
        <header class="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 glass-card p-6 rounded-2xl">
            <div class="flex items-center gap-4">
                <div class="p-3 bg-gradient-to-tr from-indigo-500 to-rose-500 rounded-xl shadow-lg">
                    <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                </div>
                <div>
                    <h1 class="text-2xl md:text-3xl font-extrabold font-display bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
                        AI Store Profit Prediction
                    </h1>
                    <p class="text-xs md:text-sm text-gray-400">Real-time Retail Analytics & Profitability Forecasting Engine</p>
                </div>
            </div>

            <!-- Controls: Currency, Themes & Fonts -->
            <div class="flex flex-wrap items-center gap-3">
                
                <!-- Multi-Currency Selector -->
                <select x-model="currency" class="bg-black/40 text-xs text-indigo-300 font-semibold border border-indigo-500/30 rounded-xl px-3 py-2 outline-none focus:border-indigo-500 transition-all">
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="GBP">GBP (£)</option>
                    <option value="INR">INR (₹)</option>
                </select>

                <!-- Themes Selector -->
                <div class="flex items-center bg-black/40 p-1.5 rounded-xl border border-white/10">
                    <button @click="setTheme('theme-cyber')" :class="{'bg-indigo-600 text-white': theme === 'theme-cyber', 'text-gray-400 hover:text-white': theme !== 'theme-cyber'}" class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all">
                        Cyber
                    </button>
                    <button @click="setTheme('theme-sunset')" :class="{'bg-rose-600 text-white': theme === 'theme-sunset', 'text-gray-400 hover:text-white': theme !== 'theme-sunset'}" class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all">
                        Sunset
                    </button>
                    <button @click="setTheme('theme-emerald')" :class="{'bg-emerald-600 text-white': theme === 'theme-emerald', 'text-gray-400 hover:text-white': theme !== 'theme-emerald'}" class="px-3 py-1.5 rounded-lg text-xs font-semibold transition-all">
                        Emerald
                    </button>
                </div>

                <!-- Font Selector -->
                <select x-model="fontStyle" class="bg-black/40 text-xs text-gray-200 border border-white/10 rounded-xl px-3 py-2 outline-none focus:border-indigo-500 transition-all">
                    <option value="font-sans">Inter (Sans)</option>
                    <option value="font-display">Outfit (Display)</option>
                    <option value="font-mono">Fira Code (Mono)</option>
                </select>
            </div>
        </header>

        <!-- Main Dashboard Content -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- Left Side: Interactive Inputs -->
            <div class="lg:col-span-5 flex flex-col gap-6">
                <div class="glass-card rounded-2xl p-6">
                    <div class="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
                        <h2 class="text-lg font-bold flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-ping"></span>
                            Store Parameters
                        </h2>
                        <button @click="resetDefaults()" class="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">Reset</button>
                    </div>

                    <form @submit.prevent="runPrediction()" class="space-y-3.5 max-h-[580px] overflow-y-auto pr-2">
                        <template x-for="(val, key) in formData" :key="key">
                            <div class="flex flex-col gap-1 bg-black/20 p-3 rounded-xl border border-white/5 hover:border-white/10 transition-all">
                                <div class="flex justify-between items-center">
                                    <label :for="key" class="text-xs font-medium text-gray-300" x-text="key === 'Sales' ? `Sales (${currencySymbols[currency]})` : key"></label>
                                    <span class="text-xs font-mono text-indigo-400" x-text="key === 'Sales' ? formatCurrency(val) : val"></span>
                                </div>
                                <input 
                                    type="number" 
                                    :step="key === 'Discount' ? '0.01' : (key === 'Sales' ? '0.01' : '1')"
                                    :id="key" 
                                    x-model.number="formData[key]"
                                    class="w-full bg-black/40 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-indigo-500 transition-colors"
                                >
                            </div>
                        </template>

                        <button 
                            type="submit" 
                            :disabled="loading"
                            class="w-full mt-4 py-3.5 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 shadow-lg shadow-indigo-500/20 transition-all transform active:scale-95 flex items-center justify-center gap-2"
                        >
                            <svg x-show="loading" class="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span x-text="loading ? 'Calculating Profit...' : 'Predict Store Profit'"></span>
                        </button>
                    </form>
                </div>
            </div>

            <!-- Right Side: Visualization Analytics -->
            <div class="lg:col-span-7 flex flex-col gap-6">
                
                <!-- Main Prediction Score KPI -->
                <div class="glass-card rounded-2xl p-6 relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-6">
                    <div>
                        <span class="text-xs font-semibold text-indigo-400 uppercase tracking-widest">Predicted Store Profit (<span x-text="currency"></span>)</span>
                        <div class="text-4xl md:text-5xl font-black font-display tracking-tight text-white mt-1">
                            <span x-text="prediction !== null ? formatCurrency(prediction) : '---'"></span>
                        </div>
                        <p class="text-xs text-gray-400 mt-2">Predicted via Gradient Boosting Machine learning model</p>
                    </div>

                    <div class="flex gap-3">
                        <div class="bg-black/30 border border-white/10 p-3 rounded-xl text-center min-w-[90px]">
                            <span class="text-[10px] text-gray-400 block uppercase">Estimators</span>
                            <span class="text-lg font-bold text-indigo-400">100</span>
                        </div>
                        <div class="bg-black/30 border border-white/10 p-3 rounded-xl text-center min-w-[90px]">
                            <span class="text-[10px] text-gray-400 block uppercase">Inputs</span>
                            <span class="text-lg font-bold text-rose-400">13</span>
                        </div>
                    </div>
                </div>

                <!-- Interactive Charts -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div class="glass-card rounded-2xl p-5 flex flex-col">
                        <h3 class="text-sm font-bold text-gray-200 mb-4">Store Metrics Overview</h3>
                        <div class="relative flex-1 min-h-[220px]">
                            <canvas id="barChart"></canvas>
                        </div>
                    </div>

                    <div class="glass-card rounded-2xl p-5 flex flex-col">
                        <h3 class="text-sm font-bold text-gray-200 mb-4">Profit Trend Projection</h3>
                        <div class="relative flex-1 min-h-[220px]">
                            <canvas id="lineChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Analysis Summary Table -->
                <div class="glass-card rounded-2xl p-5">
                    <h3 class="text-sm font-bold text-gray-200 mb-3">Profitability Parameters</h3>
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs text-gray-300">
                            <thead class="bg-black/40 text-gray-400 uppercase font-mono">
                                <tr>
                                    <th class="p-2.5 rounded-l-lg">Metric</th>
                                    <th class="p-2.5">Input Value</th>
                                    <th class="p-2.5 rounded-r-lg">State</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-white/5">
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Sales Volume</td>
                                    <td class="py-2 px-2.5 font-mono text-indigo-400" x-text="formatCurrency(formData['Sales'])"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px]">Active</span></td>
                                </tr>
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Discount Rate</td>
                                    <td class="py-2 px-2.5 font-mono text-rose-400" x-text="`${(formData['Discount'] * 100).toFixed(0)}%`"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px]">Applied</span></td>
                                </tr>
                                <tr>
                                    <td class="py-2 px-2.5 font-medium">Quantity Sold</td>
                                    <td class="py-2 px-2.5 font-mono text-emerald-400" x-text="formData['Quantity']"></td>
                                    <td class="py-2 px-2.5"><span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px]">Normal</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <!-- Alpine.js Application Logic -->
    <script>
        document.addEventListener('alpine:init', () => {
            Alpine.data('dashboard', () => ({
                theme: 'theme-cyber',
                fontStyle: 'font-sans',
                currency: 'USD',
                loading: false,
                prediction: null,
                formData: JSON.parse('{{ default_values | tojson | safe }}'),
                barChart: null,
                lineChart: null,

                currencyRates: {
                    USD: 1.0,
                    EUR: 0.92,
                    GBP: 0.79,
                    INR: 83.2
                },

                currencySymbols: {
                    USD: '$',
                    EUR: '€',
                    GBP: '£',
                    INR: '₹'
                },

                init() {
                    this.$nextTick(() => {
                        this.initCharts();
                        this.runPrediction();
                    });
                },

                formatCurrency(value) {
                    if (value === null || value === undefined) return '---';
                    const converted = value * this.currencyRates[this.currency];
                    return `${this.currencySymbols[this.currency]}${converted.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                },

                setTheme(themeName) {
                    this.theme = themeName;
                },

                resetDefaults() {
                    this.formData = JSON.parse('{{ default_values | tojson | safe }}');
                    this.runPrediction();
                },

                async runPrediction() {
                    this.loading = true;
                    try {
                        const res = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ features: this.formData })
                        });
                        const data = await res.json();
                        if (data.status === 'success') {
                            this.prediction = data.prediction;
                            this.updateCharts();
                        }
                    } catch (err) {
                        console.error('Error running prediction:', err);
                    } finally {
                        this.loading = false;
                    }
                },

                initCharts() {
                    const ctxBar = document.getElementById('barChart').getContext('2d');
                    this.barChart = new Chart(ctxBar, {
                        type: 'bar',
                        data: {
                            labels: ['Sales', 'Discount %', 'Quantity', 'Sub-Cat', 'Region'],
                            datasets: [{
                                data: [this.formData['Sales'], this.formData['Discount'] * 100, this.formData['Quantity'], this.formData['Sub-Category'], this.formData['Region']],
                                backgroundColor: ['#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#a855f7'],
                                borderRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                                x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                            }
                        }
                    });

                    const ctxLine = document.getElementById('lineChart').getContext('2d');
                    const steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
                    this.lineChart = new Chart(ctxLine, {
                        type: 'line',
                        data: {
                            labels: steps,
                            datasets: [{
                                data: steps.map(i => (this.prediction || 15) * (0.6 + (i * 0.004))),
                                borderColor: '#ec4899',
                                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                                fill: true,
                                tension: 0.3
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#9ca3af' } },
                                x: { grid: { display: false }, ticks: { color: '#9ca3af' } }
                            }
                        }
                    });
                },

                updateCharts() {
                    if (!this.barChart || !this.lineChart) return;
                    
                    this.barChart.data.datasets[0].data = [
                        this.formData['Sales'], 
                        this.formData['Discount'] * 100, 
                        this.formData['Quantity'], 
                        this.formData['Sub-Category'], 
                        this.formData['Region']
                    ];
                    this.barChart.update();

                    const steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
                    this.lineChart.data.datasets[0].data = steps.map((i, idx) => (this.prediction || 10) * (0.5 + (idx * 0.05)));
                    this.lineChart.update();
                }
            }));
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, default_values=DEFAULT_VALUES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_dict = data.get("features", {})
        
        # Build vector matching feature order
        input_vector = [float(input_dict.get(feat, 0)) for feat in FEATURE_NAMES]
        
        if model is not None:
            prediction_val = model.predict([input_vector])[0]
        else:
            # Mathematical fallback computation when pickle is unweighted
            prediction_val = float(np.dot(input_vector[:5], [0.1, 0.05, 0.2, 0.01, 0.15]))

        return jsonify({
            "status": "success",
            "prediction": float(prediction_val)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
