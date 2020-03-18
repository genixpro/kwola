import React, { Component } from 'react';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import Form from './formElements';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import Papersheet from '../../components/utility/papersheet';
import { FullColumn } from '../../components/utility/rowColumn';
import { FormsComponentWrapper, FormsMainWrapper } from './forms.style';
import CodeViewer from '../codeViewer';
import axios from "axios";

//injectTapEventPlugin();

const reducer = combineReducers({ form: reduxFormReducer });
const store = createStore(reducer);

export default class extends Component {
  state = {
    result: '',
  };

  onSubmit(values)
  {
    axios.post("/api/application", {... values}).then((response) =>
    {
        this.props.history.push(`/applications/${response.data.applicationId}`);
    });
  };

  render()
  {
    const { result } = this.state;
    return (
      <Provider store={store}>
        <LayoutWrapper>
          <FormsMainWrapper>
            <FormsComponentWrapper className=" mateFormsComponent ">
              <FullColumn>
                <Papersheet>
                  <Form onSubmit={this.onSubmit.bind(this)} />
                </Papersheet>
              </FullColumn>
            </FormsComponentWrapper>
          </FormsMainWrapper>
        </LayoutWrapper>
      </Provider>
    );
  }
}

