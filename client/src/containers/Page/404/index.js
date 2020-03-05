import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import Image from '../../../images/pages/404.png';
import IntlMessages from '../../../components/utility/intlMessages';
import FourZeroFourStyleWrapper from './404.style';

class FourZeroFour extends Component {
  render() {
    return (
      <FourZeroFourStyleWrapper className="mate404Page">
        <div className="mate404Content">
          <h1>
            <IntlMessages id="page404.title" />
          </h1>
          <h3>
            <IntlMessages id="page404.subTitle" />
          </h3>
          <p>
            <IntlMessages id="page404.description" />
          </p>

          <Link to="/dashboard">
            <button type="button">
              <IntlMessages id="page404.backButton" />
            </button>
          </Link>
        </div>

        <div className="mate404Artwork">
          <img alt="#" src={Image} />
        </div>
      </FourZeroFourStyleWrapper>
    );
  }
}

export default FourZeroFour;
