import React, { Component } from 'react';
import qs from 'qs';
import options from '../containers/Sidebar/options';

export function getInitData() {
  const initData = qs.parse(window.location.search.slice(1));
  if (initData.toggle)
    initData.toggle.free_shipping =
      initData.toggle.free_shipping === 'true' ? true : undefined;
  return initData;
}
export function setUrl(searchState) {
  const search = searchState
    ? `${window.location.pathname}?${qs.stringify(searchState)}`
    : '';
  window.history.pushState(searchState, null, search);
}

export function getDefaultPath() {
  const getParent = lastRoute => {
    const parent = {};
    if (!lastRoute) return parent;
    parent[lastRoute] = true;
    options.forEach(option => {
      if (option.children) {
        option.children.forEach(child => {
          if (child.key === lastRoute) {
            parent[option.key] = true;
          }
        });
      }
    });
    return parent;
  };
  if (window && window.location.pathname) {
    const routes = window.location.pathname.split('/');
    if (routes.length > 1) {
      const lastRoute = routes[routes.length - 1];
      return getParent(lastRoute);
    }
  }
  return [];
}
const updateAfter = 700;
const searchStateToUrl = searchState =>
  searchState ? `${window.location.pathname}?${qs.stringify(searchState)}` : '';

export const withUrlSync = App =>
  class urlSync extends Component {
    constructor() {
      super();
      this.state = { searchState: qs.parse(window.location.search.slice(1)) };
      window.addEventListener('popstate', ({ state: searchState }) =>
        this.setState({ searchState })
      );
    }

    onSearchStateChange = searchState => {
      clearTimeout(this.debouncedSetState);
      this.debouncedSetState = setTimeout(() => {
        window.history.pushState(
          searchState,
          null,
          searchStateToUrl(searchState)
        );
      }, updateAfter);
      this.setState({ searchState });
    };

    render() {
      return (
        <App
          {...this.props}
          searchState={this.state.searchState}
          onSearchStateChange={this.onSearchStateChange}
          createURL={searchStateToUrl}
        />
      );
    }
  };
