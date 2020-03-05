import React from 'react';
import AlgoliaLogo from '../../images/algolia.svg';
import { FooterWrapper } from './algoliaComponent.style';

const Footer = () => (
  <FooterWrapper className="algoliaFooter">
    <span>Powred by</span>
    <div className="logoWrapper">
      <img alt="#" src={process.env.PUBLIC_URL + AlgoliaLogo} />
    </div>
  </FooterWrapper>
);

export default Footer;
