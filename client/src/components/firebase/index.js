import React, { Component } from 'react';
import Button from '../uielements/button';
import { notification } from '../index';
import fireBtnSvg from '../../images/firebase.svg';
import Firebase from '../../helpers/firebase/index';
import { TextField, Dialog, DialogTitle } from './firebase.style';
import FirebaseLoginModal from './firebase.style';

export default class extends Component {
  state = {
    visible: false,
    email: 'demo@gmail.com',
    password: 'demodemo',
    confirmLoading: false,
  };
  showModal = () => {
    this.setState({
      visible: true,
    });
  };
  handleCancel = e => {
    this.setState({
      visible: false,
    });
  };
  handleLogin = () => {
    const { email, password } = this.state;
    if (!(email && password)) {
      notification('error', 'Please fill in email. and password');
      return;
    }
    this.setState({
      confirmLoading: true,
    });
    const self = this;
    let isError = false;
    Firebase.login(Firebase.EMAIL, { email, password })
      .catch(result => {
        const message =
          result && result.message ? result.message : 'Sorry Some error occurs';
        notification('error', message);
        self.setState({
          confirmLoading: false,
        });
        isError = true;
      })
      .then(result => {
        if (isError) {
          return;
        }
        if (!result || result.message) {
          const message =
            result && result.message
              ? result.message
              : 'Sorry Some error occurs';
          notification('error', message);
          self.setState({
            confirmLoading: false,
          });
        } else {
          self.setState({
            visible: false,
            confirmLoading: false,
          });
          this.props.login();
        }
      });
  };
  resetPassword = () => {
    const { email } = this.state;
    if (!email) {
      notification('error', `Please fill in email.`);
      return;
    }
    Firebase.resetPassword(email)
      .then(() => {
        notification('success', `Password reset email sent to ${email}.`);
        this.handleCancel();
      })
      .catch(error => notification('error', 'Email address not found.'));
  };
  render() {
    return (
      <div>
        <Button type="button" onClick={this.showModal} className="btnFirebase">
          <div className="mateLoginOtherIcon">
            <img src={fireBtnSvg} alt="Authentication Btn" />
          </div>
          <span>
            {this.props.signup
              ? 'Sign up with Firebase'
              : 'Login with Firebase'}
          </span>
        </Button>
        <Dialog
          open={this.state.visible}
          className="FirebaseLoginModal"
          onClose={this.handleCancel}
          aria-labelledby="form-dialog-title"
        >
          <DialogTitle id="form-dialog-title" className="form-dialog-title">
            <span className="form-dialog-title-text">
              Sign in with Firebase{' '}
            </span>
          </DialogTitle>
          <FirebaseLoginModal>
            <div className="inputWrapper">
              <TextField
                ref={email => (this.email = email)}
                size="large"
                label="Email"
                placeholder="Email"
                value={this.state.email}
                onChange={event => {
                  this.setState({ email: event.target.value });
                }}
              />
            </div>
            <div className="inputWrapper" style={{ marginBottom: 10 }}>
              <TextField
                type="password"
                size="large"
                placeholder="Password"
                label="Password"
                margin="normal"
                value={this.state.password}
                onChange={event => {
                  this.setState({ password: event.target.value });
                }}
              />
            </div>
            <Button className="resetPass" onClick={this.resetPassword}>
              Reset Password
            </Button>
            <Button className="firebaseCancelBtn" onClick={this.handleCancel}>
              Cancel
            </Button>
            <Button className="firebaseloginBtn" onClick={this.handleLogin}>
              Login
            </Button>
          </FirebaseLoginModal>
        </Dialog>
      </div>
    );
  }
}
