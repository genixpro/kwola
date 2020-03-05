import React, { Component } from 'react';
// import CardReactFormContainer from 'card-react';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';

import IconButton from '../uielements/iconbutton';
import TextField from '../uielements/textfield';
import Button from '../uielements/button';
import Icons from '../uielements/icon';
import notification from '../notification';
import './card.css';
import {
  CardInfoWrapper,
  InfoFormWrapper,
  DialogWrapper,
  ButtonWrapper,
} from './cardModal.style';

const theme = createMuiTheme({
  overrides: {
    MuiDialog: {
      root: {
        zIndex: 1500,
      },
    },
  },
});

export default class extends Component {
  render() {
    const {
      modalType,
      editView,
      handleCancel,
      selectedCard,
      submitCard,
      updateCard,
    } = this.props;

    this.columns = [
      {
        title: 'Number',
        dataIndex: 'number',
        key: 'number',
      },
      {
        title: 'Full Name',
        dataIndex: 'name',
        key: 'name',
      },
      {
        title: 'Expiry',
        dataIndex: 'expiry',
        key: 'expiry',
      },
      {
        title: 'CVC',
        dataIndex: 'cvc',
        key: 'cvc',
      },
      {
        title: 'Notes',
        dataIndex: 'notes',
        key: 'notes',
      },
    ];

    const saveButton = () => {
      if (!selectedCard.name) {
        notification('error', 'Please fill in name');
      } else if (!selectedCard.number) {
        notification('error', 'Please fill in card number');
      } else if (!selectedCard.expiry) {
        notification('error', 'Please fill in card expiry');
      } else if (!selectedCard.cvc) {
        notification('error', 'Please fill in card cvc');
      } else if (!selectedCard.notes) {
        notification('error', 'Please fill in card notes');
      } else {
        submitCard();
      }
    };
    const containerId = 'card-wrapper';
    // const cardConfig = {
    //   container: containerId, // required an object contain the form inputs names. every input must have a unique name prop.
    //   formTextFieldsNames: {
    //     number: 'number', // optional — default "number"
    //     expiry: 'expiry', // optional — default "expiry"
    //     cvc: 'cvc', // optional — default "cvc"
    //     name: 'name', // optional - default "name"
    //     notes: 'notes', // optional - default "notes"
    //   },
    //   initialValues: selectedCard,
    //   classes: {
    //     valid: 'valid-input', // optional — default 'jp-card-valid'
    //     invalid: 'valid-input', // optional — default 'jp-card-invalid'
    //   },
    //   formatting: true, // optional - default true
    //   placeholders: {
    //     number: '•••• •••• •••• ••••',
    //     expiry: '••/••',
    //     cvc: '•••',
    //     name: 'Full Name',
    //     notes: 'Add some notes',
    //   },
    // };
    return (
      <ThemeProvider theme={theme}>
        <DialogWrapper
          open={editView}
          className="cardOuterWrapper"
          onClose={handleCancel}
        >
          <div className="modalTitleWrapper">
            <div className="modalTitle">
              {modalType === 'edit' ? 'Edit Card' : 'Add Card'}
            </div>
            <IconButton className="modalCloseBtn" onClick={handleCancel}>
              <Icons className="modalCloseBtn" onClick={handleCancel}>
                clear
              </Icons>
            </IconButton>
          </div>

          <CardInfoWrapper id={containerId} className="cardWrapper" />

          {/*<CardReactFormContainer className="cardFormWrapper" {...cardConfig}>*/}
          <InfoFormWrapper>
            <form className="cardInfoForm">
              {this.columns.map((column, index) => {
                const { key, title } = column;
                return (
                  <TextField
                    label={title}
                    type="text"
                    className={`cardInput ${key}`}
                    onChange={event => {
                      selectedCard[key] = event.target.value;
                      updateCard(selectedCard);
                    }}
                    name={key}
                    key={index}
                  />
                );
              })}
            </form>
          </InfoFormWrapper>
          {/*</CardReactFormContainer>*/}
          <ButtonWrapper>
            <Button color="secondary" onClick={handleCancel}>
              Cancel
            </Button>

            <Button variant="contained" color="primary" onClick={saveButton}>
              {modalType === 'edit' ? 'Edit Card' : 'Add Card'}
            </Button>
          </ButtonWrapper>
        </DialogWrapper>
      </ThemeProvider>
    );
  }
}
