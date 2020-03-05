import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import Button from '../../../components/uielements/button';
import signinImg from '../../../images/signup.svg';
import TextField from '../../../components/uielements/textfield';
import IntlMessages from '../../../components/utility/intlMessages';
import SignInStyleWrapper from './forgotPassword.style';

class ForgotPassword extends Component {
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
              <IntlMessages id="page.forgetPassSubTitle" />
            </h1>
            <p>
              <IntlMessages id="page.forgetPassDescription" />
            </p>
          </div>
          <div className="mateSignInPageForm">
            <div className="mateInputWrapper">
              <TextField label="Enter your email" margin="normal" />
            </div>
            <div className="mateLoginSubmit">
              <Button type="button">
                <IntlMessages id="page.sendRequest" />
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

export default ForgotPassword;
