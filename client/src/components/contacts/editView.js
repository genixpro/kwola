import React, { Component } from 'react';
import { createMuiTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import Button from '../uielements/button';
import Scrollbars from '../utility/customScrollBar';
import { notification } from '../index';
import {
  ContactModal,
  PersonNameImg,
  Avatar,
  ContactViewModal,
  ModalView,
  ButtonGroup,
  IconButton,
  InputSearch,
} from './contact.style';

const theme = createMuiTheme({
  overrides: {
    MuiDialog: {
      root: {
        zIndex: 1500,
      },
      paperWidthSm: {
        width: '100%',
        margin: 15,
      },
    },
  },
});

const getContact = (seletedContact, otherAttributes) => {
  const contact = {
    id: seletedContact.id,
    avatar: seletedContact.avatar,
  };
  otherAttributes.forEach(attr => {
    contact[attr.value] = seletedContact[attr.value];
  });
  return contact;
};
export default class extends Component {
  state = {
    contact: getContact(this.props.seletedContact, this.props.otherAttributes),
    isFresh: this.props.seletedContact.isFresh,
    view: this.props.seletedContact.isFresh ? 'edit' : 'display',
  };
  showModal = () => {
    this.props.setSelectedContact();
  };
  handleCancel = e => {
    this.props.setSelectedContact();
  };
  handleDelete = e => {
    const { contacts, updateContacts } = this.props;
    const { contact } = this.state;
    const newContacts = [];
    contacts.forEach(singleContact => {
      if (contact.id !== singleContact.id) {
        newContacts.push(singleContact);
      }
    });
    notification('error', `${contact.name || 'No Name'} Detelted`);
    updateContacts(newContacts);
  };
  handleSave = () => {
    const { contacts, updateContacts } = this.props;
    const { contact, isFresh } = this.state;
    const newContacts = [];
    contacts.forEach(singleContact => {
      if (contact.id === singleContact.id) {
        newContacts.push(contact);
      } else {
        newContacts.push(singleContact);
      }
    });
    if (isFresh) {
      newContacts.push(contact);
    }

    updateContacts(newContacts);
  };
  EditView = () => {
    const { contact, isFresh } = this.state;
    const { otherAttributes } = this.props;
    return (
      <ModalView>
        <Scrollbars style={{ maxHeight: '60vh' }}>
          {otherAttributes.map(attr => (
            <div className="contactInfo" key={attr.value}>
              <InputSearch
                alwaysDefaultValue
                onChange={value => {
                  contact[attr.value] = value;
                  this.setState({ contact });
                }}
                defaultValue={contact[attr.value]}
                placeholder={attr.title}
              />
            </div>
          ))}
        </Scrollbars>

        <ButtonGroup>
          {isFresh ? (
            ''
          ) : (
            <Button color="secondary" onClick={this.handleDelete}>
              Delete
            </Button>
          )}
          <IconButton onClick={this.handleCancel}>clear</IconButton>
          <Button color="primary" onClick={this.handleSave}>
            Save
          </Button>
        </ButtonGroup>
      </ModalView>
    );
  };
  DetailView = () => {
    const { contact, isFresh } = this.state;
    const { otherAttributes } = this.props;
    return (
      <ModalView>
        <Scrollbars style={{ maxHeight: '60vh' }}>
          {otherAttributes.map(attr =>
            contact[attr.value] ? (
              <div className="contactInfo" key={attr.value}>
                <h6>{attr.value}</h6>
                <span>{contact[attr.value]}</span>
              </div>
            ) : (
              <div key={attr.value} />
            )
          )}
        </Scrollbars>
        <ButtonGroup>
          {isFresh ? (
            ''
          ) : (
            <Button color="secondary" onClick={this.handleDelete}>
              Delete
            </Button>
          )}
          <IconButton onClick={this.handleCancel}>clear</IconButton>
          <Button
            color="primary"
            onClick={() => this.setState({ view: 'edit' })}
          >
            Edit
          </Button>
        </ButtonGroup>
      </ModalView>
    );
  };
  render() {
    const { contact, view } = this.state;
    return (
      <ThemeProvider theme={theme}>
        <ContactViewModal
          title="Contact"
          open={true}
          className="contactViewModal"
          onClose={this.handleCancel}
        >
          <ContactModal className={view === 'edit' ? 'editView' : ''}>
            <PersonNameImg>
              <label htmlFor="inputUpload">
                <Avatar alt={contact.name} src={contact.avatar} />
                {view === 'edit' ? (
                  <input
                    type="file"
                    id="inputUpload"
                    name="inputUpload"
                    className="inputUpload"
                  />
                ) : (
                  ''
                )}
              </label>
              {view === 'edit' ? '' : <h2>{contact.name || 'No Name'}</h2>}
            </PersonNameImg>
            {view === 'edit' ? <this.EditView /> : <this.DetailView />}
          </ContactModal>
        </ContactViewModal>
      </ThemeProvider>
    );
  }
}
