import Auth0Lock from 'auth0-lock';
import history from './history';
import { Auth0Config } from '../../settings';
import { notification } from '../../components';

class Auth0Helper {
  isValid = Auth0Config.clientID && Auth0Config.domain;

  constructor() {
    this.login = this.login.bind(this);
    this.logout = this.logout.bind(this);
    this.handleAuthentication = this.handleAuthentication.bind(this);
    this.isAuthenticated = this.isAuthenticated.bind(this);
  }
  login(handleLogin) {
    this.lock = this.isValid
      ? new Auth0Lock(
          Auth0Config.clientID,
          Auth0Config.domain,
          Auth0Config.options
        )
      : null;
    if (!this.lock) {
      notification('error', 'Lock Error');
    }
    this.lock.on('authenticated', authResult => {
      if (authResult && authResult.accessToken) {
        if (window) {
          localStorage.setItem('id_token', authResult.accessToken);
        }
        handleLogin();
      } else {
        notification('error', 'Wrong mail or password');
      }
    });
    this.lock.show();
  }
  handleAuthentication(props) {
    localStorage.setItem('id_token', 'secret token');
    history.replace('/dashboard');
  }
  setSession(authResult) {
    // Set the time that the access token will expire at
    let expiresAt = JSON.stringify(
      authResult.expiresIn * 1000 + new Date().getTime()
    );
    localStorage.setItem('access_token', authResult.accessToken);
    localStorage.setItem('id_token', authResult.idToken);
    localStorage.setItem('expires_at', expiresAt);
    // navigate to the home route
    history.replace('/');
  }

  logout() {
    // Clear access token and ID token from local storage
    localStorage.removeItem('access_token');
    localStorage.removeItem('id_token');
    localStorage.removeItem('expires_at');
    // navigate to the home route
    history.replace('/');
  }

  isAuthenticated() {
    // Check whether the current time is past the
    // access token's expiry time
    return (
      new Date().getTime() < JSON.parse(localStorage.getItem('expires_at'))
    );
  }
}
export default new Auth0Helper();
