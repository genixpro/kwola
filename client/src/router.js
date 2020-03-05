import React, { lazy } from 'react';
import { Route, Redirect } from 'react-router-dom';
import { connect } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import App from './containers/App';
import Auth0 from './helpers/auth0';

const RestrictedRoute = ({ component: Component, isLoggedIn, ...rest }) => (
  <Route
    {...rest}
    render={props =>
      isLoggedIn ? (
        <Component {...props} />
      ) : (
        <Redirect
          to={{
            pathname: '/signin',
            state: { from: props.location },
          }}
        />
      )
    }
  />
);

const PublicRoutes = ({ history, isLoggedIn }) => (
  <BrowserRouter>
    <>
      <Route
        exact
        path="/"
        component={lazy(() => import('./containers/Page/signin'))}
      />
      <Route
        exact
        path="/signin"
        component={lazy(() => import('./containers/Page/signin'))}
      />
      <Route
        path="/auth0loginCallback"
        render={props => {
          Auth0.handleAuthentication(props);
        }}
      />
      <RestrictedRoute
        path="/dashboard"
        component={App}
        isLoggedIn={isLoggedIn}
      />
      <Route
        exact
        path="/404"
        component={lazy(() => import('./containers/Page/404'))}
      />
      <Route
        exact
        path="/505"
        component={lazy(() => import('./containers/Page/505'))}
      />
      <Route
        exact
        path="/signup"
        component={lazy(() => import('./containers/Page/signup'))}
      />
      <Route
        exact
        path="/forgot-password"
        component={lazy(() => import('./containers/Page/forgetpassword'))}
      />
      <Route
        exact
        path="/reset-password"
        component={lazy(() => import('./containers/Page/resetpassword'))}
      />
    </>
  </BrowserRouter>
);

function mapStateToProps(state) {
  return {
    isLoggedIn: state.Auth.idToken !== null,
  };
}
export default connect(mapStateToProps)(PublicRoutes);
