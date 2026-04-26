import streamlit as st

# Default Variables
lr_pv = 23
ref_price_pv = 0.08
lr_bat = 15
ref_price_bat = 110

# Sidebar for User Input
st.sidebar.title("User Input")

# User Input with Default Initialization
selected_lr_pv = st.sidebar.number_input('Learning Rate PV', value=lr_pv)
selected_ref_price_pv = st.sidebar.number_input('Reference Price PV', value=ref_price_pv)
selected_lr_bat = st.sidebar.number_input('Learning Rate BAT', value=lr_bat)
selected_ref_price_bat = st.sidebar.number_input('Reference Price BAT', value=ref_price_bat)

# Enhanced CSS Styles
st.markdown("<style>", unsafe_allow_html=True)
st.markdown("body { background: linear-gradient(to right, #f0f8ff, #e0ffff); font-family: 'Arial', sans-serif; color: #333; }", unsafe_allow_html=True)
st.markdown("h1 { color: #003366; }", unsafe_allow_html=True)
st.markdown(".stButton { background-color: #4CAF50; color: white; }", unsafe_allow_html=True)
st.markdown("</style>", unsafe_allow_html=True)

# Sample Function
def calculate_price(lr, ref_price):
    return lr * ref_price

# Display Results
if st.button('Calculate'):
    result = calculate_price(selected_lr_pv, selected_ref_price_pv)
    st.write(f'Calculated Price PV: {result}')

# Optionally, add more functionalities and clean layout here.

# Fixes applied:
# 1. Initialized sidebar variables with default values before conditionals to fix NameError.
# 2. Improved color contrast in CSS for better visibility.
# 3. Simplified UI text for clarity.
# 4. Removed text truncations in markdown strings.
# 5. Ensured all functions and data structures are maintained.