import React, { Component } from 'react';
import MailPagination from './mailPagination.style';
import { rtl } from '../../../settings/withDirection';

export default class PaginationControl extends Component {
  render() {
    return (
      <MailPagination className="mailPagination">
        <button type="button" className="prevPage">
          <i
            className={
              rtl === 'rtl' ? 'ion-ios-arrow-forward' : 'ion-ios-arrow-back'
            }
          />
        </button>

        <button type="button" className="nextPage">
          <i
            className={
              rtl === 'rtl' ? 'ion-ios-arrow-back' : 'ion-ios-arrow-forward'
            }
          />
        </button>
      </MailPagination>
    );
  }
}
