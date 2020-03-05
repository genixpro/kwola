import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import Button from '../../../components/uielements/button';
import IntlMessages from '../../../components/utility/intlMessages';
import signinImg from '../../../images/signup.svg';
import TextField from '../../../components/uielements/textfield';
import SignInStyleWrapper from './resetPassword.style';

class ResetPassword extends Component {
  render() {
    return (
      <SignInStyleWrapper className="mateSignInPage">
        <div className="mateSignInPageImgPart">
          <div className="mateSignInPageImg">
            <img src={signinImg} alt="Kiwi standing on oval" />
          </div>
        </div>

        <div className="mateSignInPageContent">
          <div className="mateSignInPageGreet">
            <h1>
              <IntlMessages id="page.resetPassSubTitle" />
            </h1>
            <p>
              <IntlMessages id="page.resetPassDescription" />
            </p>
          </div>
          <div className="mateSignInPageForm">
            <div className="mateInputWrapper">
              <TextField label="Enter new password" margin="normal" />
            </div>

            <div className="mateInputWrapper">
              <TextField label="Confirm password" margin="normal" />
            </div>

            <div className="mateLoginSubmit">
              <Button type="button">
                <IntlMessages id="page.resetPassSave" />
              </Button>
            </div>
          </div>

          <p className="homeRedirection">
            Or go back to{' '}
            <Link to="/dashboard">
              <Button color="primary">Homepage</Button>
            </Link>
          </p>
        </div>
      </SignInStyleWrapper>
    );
  }
}

export default ResetPassword;
