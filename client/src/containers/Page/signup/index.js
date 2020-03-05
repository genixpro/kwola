import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import { connect } from 'react-redux';
import signinImg from '../../../images/signin.svg';
import fbBtnSvg from '../../../images/facebook-app-symbol.svg';
import gpBtnSvg from '../../../images/google-plus.svg';
import authBtnSvg from '../../../images/auth0.svg';
import TextField from '../../../components/uielements/textfield';
import Scrollbars from '../../../components/utility/customScrollBar';
import Button from '../../../components/uielements/button';
import authAction from '../../../redux/auth/actions';
import IntlMessages from '../../../components/utility/intlMessages';
import SignUpStyleWrapper from './signup.style';
import Auth0 from '../../../helpers/auth0/index';
import Firebase from '../../../helpers/firebase';
import FirebaseLogin from '../../../components/firebase';
import { Checkbox } from './signup.style';

const { login } = authAction;

class SignUp extends Component {
  state = {
    redirectToReferrer: false,
  };
  componentWillReceiveProps(nextProps) {
    if (
      this.props.isLoggedIn !== nextProps.isLoggedIn &&
      nextProps.isLoggedIn === true
    ) {
      this.setState({ redirectToReferrer: true });
    }
  }
  handleLogin = () => {
    const { login } = this.props;
    login();
    this.props.history.push('/dashboard');
  };
  render() {
    return (
      <SignUpStyleWrapper className="mateSignUpPage">
        <div className="mateSignInPageImgPart">
          <div className="mateSignInPageImg">
            <img src={signinImg} alt="Kiwi standing on oval" />
          </div>
        </div>

        <div className="mateSignInPageContent">
          <div className="mateSignInPageLink">
            <Link to="#">
              <button className="mateSignInPageLinkBtn active" type="button">
                Register
              </button>
            </Link>
            <Link to="/signin">
              <button className="mateSignInPageLinkBtn " type="button">
                Login
              </button>
            </Link>
          </div>
          <Scrollbars style={{ height: '100%' }}>
            <div className="mateSignInPageGreet">
              <h1>Its Free, Join Us</h1>
              <p>
                Welcome to Mate Admin, Please SignUp with your personal account
                information.
              </p>
            </div>
            <div className="mateSignInPageForm">
              <div className="mateInputWrapper">
                <TextField
                  label="Username"
                  placeholder="Username"
                  margin="normal"
                />
              </div>
              <div className="mateInputWrapper">
                <TextField
                  label="Email"
                  placeholder="Email"
                  margin="normal"
                  type="Email"
                />
              </div>
              <div className="mateInputWrapper">
                <TextField
                  label="Password"
                  placeholder="Password"
                  margin="normal"
                  type="Password"
                />
              </div>
              <div className="mateInputWrapper">
                <TextField
                  label="Confirm Password"
                  placeholder="Confirm Password"
                  margin="normal"
                  type="Password"
                />
              </div>
            </div>
            <div className="mateAgreement">
              <div className="mateLoginSubmitCheck">
                <Checkbox color="primary" className="mateTermsCheck" />
                <span className="mateTermsText">
                  <IntlMessages id="page.signUpTermsConditions" />
                </span>
              </div>
              <div className="mateLoginSubmit">
                <Button type="button" onClick={this.handleLogin}>
                  Sign Up
                </Button>
              </div>
            </div>
            <div className="mateLoginSubmitText">
              <span>or Sign Up with </span>
            </div>
            <div className="mateLoginOtherBtn">
              <div className="mateLoginOtherBtnWrap">
                <Button
                  onClick={this.handleLogin}
                  type="button"
                  className="btnFacebook"
                >
                  <div className="mateLoginOtherIcon">
                    <img src={fbBtnSvg} alt="facebook Btn" />
                  </div>
                  <IntlMessages id="page.signUpFacebook" />
                </Button>
              </div>
              <div className="mateLoginOtherBtnWrap">
                <Button
                  onClick={this.handleLogin}
                  type="button"
                  className="btnGooglePlus"
                >
                  <div className="mateLoginOtherIcon">
                    <img src={gpBtnSvg} alt="Google Plus Btn" />
                  </div>
                  <IntlMessages id="page.signUpGooglePlus" />
                </Button>
              </div>
              <div className="mateLoginOtherBtnWrap">
                {Auth0.isValid ? (
                  <Button
                    type="button"
                    className="btnAuthZero"
                    onClick={() => {
                      Auth0.login(this.handleLogin);
                    }}
                  >
                    <div className="mateLoginOtherIcon">
                      <img src={authBtnSvg} alt="Authentication Btn" />
                    </div>
                    <IntlMessages id="page.signUpAuth0" />
                  </Button>
                ) : (
                  <Button
                    type="button"
                    className="btnAuthZero"
                    onClick={this.handleLogin}
                  >
                    <div className="mateLoginOtherIcon">
                      <img src={authBtnSvg} alt="Authentication Btn" />
                    </div>
                    <IntlMessages id="page.signUpAuth0" />
                  </Button>
                )}
              </div>
              <div className="mateLoginOtherBtnWrap">
                {Firebase.isValid && (
                  <FirebaseLogin signup={true} login={this.handleLogin} />
                )}
              </div>
            </div>
          </Scrollbars>
        </div>
      </SignUpStyleWrapper>
    );
  }
}

export default connect(
  state => ({
    isLoggedIn: state.Auth.idToken !== null ? true : false,
  }),
  { login }
)(SignUp);
