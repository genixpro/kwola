import React, { Component } from 'react';
import Async from '../../../helpers/asyncComponent';
import Input from '../../uielements/input';
import ComposeAutoComplete from '../composeAutoComplete';
import notification from '../../notification';
import IntlMessages from '../../utility/intlMessages';
import ComposeForm, {
  EditorWrapper,
  Button,
  IconButton,
  Icon,
} from './composeMail.style';

const Editor = props => (
  <Async
    load={import(
      /* webpackChunkName: "compose-mAIL--editor" */ '../../uielements/editor'
    )}
    componentProps={props}
  />
);

function uploadCallback(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'https://api.imgur.com/3/image');
    xhr.setRequestHeader('Authorization', 'Client-ID 8d26ccd12712fca');
    const data = new FormData();
    data.append('image', file);
    xhr.send(data);
    xhr.addEventListener('load', () => {
      const response = JSON.parse(xhr.responseText);
      resolve(response);
    });
    xhr.addEventListener('error', () => {
      const error = JSON.parse(xhr.responseText);
      reject(error);
    });
  });
}
export default class ComposeMail extends Component {
  constructor(props) {
    super(props);
    this.state = {
      editorState: null,
      loading: false,
      iconLoading: false,
    };
  }
  render() {
    const onEditorStateChange = editorState => {
      this.setState({ editorState });
    };
    const ComposeAutoCompleteTO = {
      allMails: this.props.allMails,
      updateData: () => {},
      placeholder: 'To',
      autofocus: true,
    };
    const ComposeAutoCompleteCC = {
      allMails: this.props.allMails,
      updateData: () => {},
      placeholder: 'CC',
    };
    const editorOption = {
      style: { width: '90%', height: '70%' },
      editorState: this.state.editorState,
      toolbarClassName: 'home-toolbar',
      wrapperClassName: 'home-wrapper',
      editorClassName: 'home-editor',
      onEditorStateChange: onEditorStateChange,
      uploadCallback: uploadCallback,
      toolbar: { image: { uploadCallback: uploadCallback } },
    };

    const { singleMail, placeholder } = this.props;
    const placeholderText = placeholder ? placeholder : 'Say something';

    return (
      <ComposeForm className={this.props.className}>
        {!singleMail ? <ComposeAutoComplete {...ComposeAutoCompleteTO} /> : ''}
        {!singleMail ? <ComposeAutoComplete {...ComposeAutoCompleteCC} /> : ''}
        {!singleMail ? (
          <Input
            disableUnderline
            fullWidth
            placeholder="Subject"
            className="suBj3cT"
          />
        ) : (
          ''
        )}
        <EditorWrapper className="eDit0R-wRapP3r">
          <Editor
            placeholder={placeholderText}
            className="mailComposeEditor"
            {...editorOption}
          />
          <Button
            className="sEnd-bTn"
            size="small"
            onClick={e => notification('success', `Mail has been sent`, '')}
            variant="contained"
          >
            <IntlMessages id="email.send" />
          </Button>

          <IconButton
            className="dEl3Te-bTn"
            onClick={() => {
              this.props.changeComposeMail(false);
            }}
          >
            <Icon>delete</Icon>
          </IconButton>
        </EditorWrapper>
      </ComposeForm>
    );
  }
}
