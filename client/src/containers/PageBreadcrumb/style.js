import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icon from '../../components/uielements/icon/index.js';
import { boxShadow } from '../../settings/style-util';

const FakeBoxWithTab = styled.div`
  padding-bottom: 120px;

  @media only screen and (max-width: 767px) {
    padding-bottom: 127px;
  }
`;

const FakeBox = styled.div`
  padding-bottom: 70px;

  @media only screen and (max-width: 767px) {
    padding-bottom: 90px;
  }
`;

const Icons = styled(Icon)`
  color: #fff;
  font-size: 30px;
`;

const NavLinks = styled.div`
  display: flex;
  margin-top: 35px;
  margin-left: -15px;

  @media only screen and (max-width: 767px) {
    margin-bottom: 0;
    margin-top: 25px;
  }

  a {
    display: inline-flex;
    font-weight: 500;
    padding: 0 15px 0;
    color: #ffffff;
    line-height: 1.1;
    font-size: 14px;
    position: relative;
    margin-right: 15px;
    text-decoration: none;

    @media only screen and (max-width: 767px) {
      font-weight: 400;
    }

    &:last-child {
      margin-right: 0;
    }

    &.active {
      @media only screen and (max-width: 767px) {
        font-weight: 700;
      }
      &:after {
        content: '';
        width: 100%;
        height: 2px;
        display: inline-flex;
        background-color: #ffffff;
        position: absolute;
        bottom: -17px;
        left: 0;

        @media only screen and (max-width: 767px) {
          ${'' /* display: none; */};
        }
      }
    }
  }
`;

const PageInfo = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;

  .pageTitle {
    font-size: 24px;
    color: #ffffff;
    font-weight: 500;
    line-height: 1.5;
    margin-top: 0;
    margin-bottom: 0;
  }
`;

const NavRoutes = styled.div`
  flex-shrink: 0;
  position: absolute;
  bottom: 16px;
  right: 20px;

  @media only screen and (max-width: 767px) {
    position: relative;
    bottom: auto;
    right: auto;
  }

  a,
  span {
    display: inline-flex;
    padding: 0;
    color: #ffffff;
    line-height: 1.1;
    font-size: 14px;
    position: relative;
    margin-right: 5px;
    text-decoration: none;

    &:last-child {
      margin-right: 0;
    }
  }

  a {
    &:after {
      content: '/';
      padding-left: 5px;
    }
  }
`;

const BreadcrumbWrapper = styled.div`
  background: ${palette('indigo', 4)};
  padding: 17px 30px;
  width: 100%;
  position: absolute;
  top: 0;
  z-index: 1;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  ${boxShadow('0 1px 2px rgba(0,0,0,0.35)')};

  * {
    box-sizing: border-box;
  }

  @media only screen and (max-width: 767px) {
    flex-direction: column;
    padding: 15px 20px 17px;
  }
`;

export default BreadcrumbWrapper;
export { Icons, NavRoutes, NavLinks, PageInfo, FakeBoxWithTab, FakeBox };
