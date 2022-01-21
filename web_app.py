import pandas as pd
import streamlit as st
from scripts.create_splits import main as create_splits_main
from scripts.train import main as train_main
from scripts.test import main as test_main
from scripts.save_last_train import main as save_model_main
from tensorflow.python.framework.errors import NotFoundError


st.title('Image Classification APP')

st.sidebar.title('options')
radio_button = st.sidebar.radio('select', ['Create splits', 'Train', 'Test'])
save_button = st.sidebar.button('Save Trained Model')

if radio_button == 'Create splits':
    # TODO - include dataset_path, splits_dest_path, seed
    # TODO - show stats from the dataset splits and the full dataset
    st.write('''
    Use these parameters to split your data into **train**, **test** and **validation** splits
    
    If you do not want to change the parameters, just click "**Create splits**"
    ''')
    sampling_type = None
    sampling_ratio = None
    col1, col2 = st.columns(2)
    with col1:
        sampling_check = st.checkbox("Sampling")
        if sampling_check:
            sampling_type = st.radio("Sampling", ['undersample', 'oversample'])
            # TODO - get min sampling ratio based on the dataset or show the exception to the user ("please raise/under the ratio")
            sampling_ratio = st.slider(sampling_type + " ratio %", value=50, min_value=1, max_value=100) / 100.

    with col2:
        with st.form("create_splits"):
            test_size = st.slider("Test size %", value=15, min_value=1, max_value=80)/100.
            valid_size = st.slider("Validation size %", value=15, min_value=1, max_value=80)/100.

            submitted = st.form_submit_button("Create splits")
            if submitted:
                st.write("Running script **create_splits.py**")
                if sampling_ratio is not None:
                    if sampling_type == 'undersample':
                        create_splits_main(test_size=test_size, valid_size=valid_size, undersample_ratio=sampling_ratio,
                                           streamlit_callbacks=(st.write, st.progress))
                    elif sampling_type == 'oversample':
                        create_splits_main(test_size=test_size, valid_size=valid_size, oversample_ratio=sampling_ratio,
                                           streamlit_callbacks=(st.write, st.progress))
                else:
                    create_splits_main(test_size=test_size, valid_size=valid_size, streamlit_callbacks=(st.write,
                                                                                                        st.progress))
                st.write("Finished running **create_splits.py** ✔️")

elif radio_button == 'Train':
    # TODO - include train_path, valid_path
    # TODO - delete before new training
    st.write('''
        Use these parameters to **train your neural network**
        
        If you do not want to change the parameters, just click "**Train**"
        ''')
    with st.form("train"):
        img_height = st.slider("Image height", value=224, min_value=10, max_value=1000)
        img_width = st.slider("Image width", value=224, min_value=10, max_value=1000)
        batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32, 64, 128, 256, 512, 1028], value=64)
        n_hidden = st.select_slider("Number of hidden neurons", options=[64, 128, 256, 512, 1028, 2056], value=512)
        base_lr = st.number_input("Learning rate of the first learning stage", value=0.001, step=0.001, format="%.5f")
        fine_tuning_lr = st.number_input("Learning rate of the second learning stage (fine tuning)", value=0.001,
                                         step=0.001, format="%.5f")
        base_epochs = st.number_input("Number of epochs during the first learning stage", value=30, step=1, min_value=1)
        fine_tuning_epochs = st.number_input("Number of epochs during the second learning stage (fine tuning)",
                                             value=30, step=1, min_value=1)

        submitted = st.form_submit_button("Train")
        if submitted:
            st.write("Running script **train.py**")
            train_main(img_height=img_height, img_width=img_width, batch_size=batch_size, n_hidden=n_hidden,
                       base_lr=base_lr, fine_tuning_lr=fine_tuning_lr, base_epochs=base_epochs,
                       fine_tuning_epochs=fine_tuning_epochs, streamlit_callbacks=(st.write, st.progress))
            st.write("Finished running **train.py** ✔️")

elif radio_button == 'Test':
    # TODO - include train_path, valid_path
    # TODO - check if trained
    st.write('''
        Use these parameters to **test your neural network**
        
        If you do not want to change the parameters, just click "**Test**"
        ''')
    with st.form("test"):
        img_height = st.slider("Image height", value=224, min_value=10, max_value=1000)
        img_width = st.slider("Image width", value=224, min_value=10, max_value=1000)
        batch_size = st.select_slider("Batch size", options=[4, 8, 16, 32, 64, 128, 256, 512, 1028], value=64)
        n_hidden = st.select_slider("Number of hidden neurons", options=[64, 128, 256, 512, 1028, 2056], value=512)

        submitted = st.form_submit_button("Test")
        if submitted:
            st.write("Running script **test.py**")
            classification_report_dict, df_confusion_matrix = test_main(batch_size=batch_size, img_height=img_height,
                                                                          img_width=img_width, n_hidden=n_hidden,
                                                                          return_results=True)
            st.write(pd.DataFrame(classification_report_dict).T)
            st.write(df_confusion_matrix)
            st.write("Finished running **test.py** ✔️")

if save_button:
    try:
        save_model_main()
        st.sidebar.write('**MODEL SAVED** ✔️')
    except NotFoundError:
        st.sidebar.write("**DIDN'T FIND THE TRAINED MODEL FILE**")
