import streamlit as st
import subprocess
from pathlib import Path
from utils.data import get_platform_shell

st.title('Image Classification APP')

st.sidebar.title('options')
radio_button = st.sidebar.radio('select', ['Create splits', 'Train', 'Test'])

if radio_button == 'Create splits':
    # TODO - include dataset_path, splits_dest_path, seed
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
            sampling_ratio = st.slider(sampling_type + " ratio %", value=50, min_value=1, max_value=100) / 100.

    with col2:
        with st.form("create_splits"):
            test_size = st.slider("Test size %", value=15, min_value=1, max_value=80)/100.
            valid_size = st.slider("Validation size %", value=15, min_value=1, max_value=80)/100.

            submitted = st.form_submit_button("Create splits")
            if submitted:
                expression = ['python', str(Path(r'./scripts/create_splits.py')), '--test_size', str(test_size),
                              '--valid_size', str(valid_size)]
                if sampling_ratio is not None:
                    expression = expression + ['--' + str(sampling_type) + '_ratio', str(sampling_ratio)]
                st.write("Running script **create_splits.py**")
                st.write("Expression used: " + ' '.join(expression))
                st.write("...")
                p = subprocess.run(expression, shell=get_platform_shell(), check=True)
                # print('expression ---------------', expression) # for debug
                st.write("Finished running **create_splits.py** ✔️")

elif radio_button == 'Train':
    # TODO - include train_path, valid_path
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
            expression = ['python', str(Path(r'./scripts/train.py')), '--img_height', str(img_height),
                          '--img_width', str(img_width), '--batch_size', str(batch_size), '--n_hidden',
                          str(n_hidden), '--base_lr', str(base_lr), '--fine_tuning_lr', str(fine_tuning_lr)]
            st.write("Running script **train.py**")
            st.write("Expression used: " + ' '.join(expression))
            st.write("...")
            p = subprocess.run(expression, shell=get_platform_shell(), check=True)
            # print('expression ---------------', expression) # for debug
            st.write("Finished running **train.py** ✔️")

elif radio_button == 'Test':
    # TODO - improve results output style in streamlit
    # TODO - include train_path, valid_path
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
            expression = ['python', str(Path(r'./scripts/test.py')), '--img_height', str(img_height),
                          '--img_width', str(img_width), '--batch_size', str(batch_size), '--n_hidden',
                          str(n_hidden)]
            st.write("Running script **test.py**")
            st.write("Expression used: " + ' '.join(expression))
            st.write("...")
            p = subprocess.run(expression, shell=get_platform_shell(), check=True, stdout=subprocess.PIPE)
            results_text = p.stdout.decode()
            results_text = results_text[results_text.find('%') + 1:]
            st.write("Results:")
            st.write(results_text)
            st.write("Finished running **test.py** ✔️")
