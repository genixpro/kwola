import React from 'react';
import { Link } from 'react-router-dom';
import Image from '../../../images/pages/404.png';
import IntlMessages from '../../../components/utility/intlMessages';
import FiveZeroFiveStyleWrapper from './505.style';

class FiveHundredFive extends React.Component {
  render() {
    return (
      <FiveZeroFiveStyleWrapper className="mate500Page">
        <div className="mate500Content">
          <h1>
            <IntlMessages id="page500.title" />
          </h1>
          <h3>
            <IntlMessages id="page500.subTitle" />
          </h3>
          <p>
            <IntlMessages id="page500.description" />
          </p>
          <Link to="/dashboard">
            <button type="button">
              <IntlMessages id="page500.backButton" />
            </button>
          </Link>
        </div>

        <div className="mate500Artwork">
          <img alt="#" src={Image} />
        </div>
      </FiveZeroFiveStyleWrapper>
    );
  }
}

export default FiveHundredFive;
