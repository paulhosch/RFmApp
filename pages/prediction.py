import streamlit as st
import geemap.ml as ml
import ee

def predict():
    st.title("Predict")

    # Check if necessary data is in session state
    if 'logo_folds' not in st.session_state or 'outer_models' not in st.session_state:
        st.error("Please run model training and evaluation first.")
        return

    # Get observation group labels
    observation_group_labels = [group.get('label', f"Group {i}") for i, group in enumerate(st.session_state.logo_folds)]

    # Create selectbox for observation group
    selected_group_label = st.selectbox("Select Observation Group to Predict", observation_group_labels)

    # Find the fold index for the selected group
    fold_index = next((i for i, fold in enumerate(st.session_state.logo_folds) if fold.get('label') == selected_group_label), None)

    if fold_index is None:
        st.error(f"No fold found for the selected group: {selected_group_label}")
        return

    # Get the corresponding model
    model = st.session_state.outer_models[fold_index]

    # Get input features
    input_features = st.session_state.get('selected_features', [])

    if not input_features:
        st.error("No input features found. Please select features in the feature selection page.")
        return

    # Create input fields for each feature
    st.subheader("Enter Feature Values")
    input_values = {}
    for feature in input_features:
        input_values[feature] = st.number_input(f"Enter value for {feature}", value=0.0, format="%.6f")

    if st.button("Predict"):
        # Convert the model to a string representation
        model_str_list = ml.rf_to_strings(model, input_features)

        # Convert the string representation to an Earth Engine classifier
        model_ee = ml.strings_to_classifier(model_str_list)

        # Create an Earth Engine feature with the input values
        ee_feature = ee.Feature(None, {feature: value for feature, value in input_values.items()})

        # Make the prediction
        prediction = model_ee.classify(ee_feature).getInfo()

        st.success(f"Prediction: {prediction}")

        # Display additional information
        st.subheader("Model Information")
        st.write(f"Fold Index: {fold_index}")
        st.write(f"Test Group: {selected_group_label}")
        st.write("Input Features:")
        st.json(input_values)

    # Optionally, display the selected group's data
    if st.checkbox("Show Selected Group Data"):
        selected_group = st.session_state.logo_folds[fold_index]
        st.subheader(f"Data for {selected_group_label}")
        st.write("Training Data Shape:", selected_group['X_train'].shape)
        st.write("Testing Data Shape:", selected_group['X_test'].shape)
        st.write("Group Date:", selected_group.get('date', 'Not available'))
        # Add any other relevant information from the group
