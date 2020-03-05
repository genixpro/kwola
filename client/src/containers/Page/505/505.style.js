import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius } from '../../../settings/style-util';
import WithDirection from '../../../settings/withDirection';

const FiveZeroFiveStyleWrapper = styled.div`
  width: 100%;
  height: 100vh;
  display: flex;
  flex-flow: row wrap;
  align-items: center;
  justify-content: center;
  position: relative;
  box-sizing: border-box;

  * {
    box-sizing: border-box;
  }

  @media only screen and (max-width: 767px) {
    flex-direction: column;
    flex-wrap: nowrap;
    padding: 100px 15px;
    height: auto;
  }

  .mate500Content {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    flex-direction: column;

    @media only screen and (max-width: 767px) {
      order: 2;
      margin-top: 20px;
      align-items: center;
      text-align: center;
    }

    h1 {
      font-size: 84px;
      font-weight: 700;
      color: ${palette('pages', 0)};
      line-height: 1;
      margin: 0 0 25px;
    }

    h3 {
      font-size: 24px;
      font-weight: 400;
      color: ${palette('pages', 1)};
      margin: 0 0 10px;
      line-height: 1.2;
    }

    p {
      font-size: 14px;
      font-weight: 400;
      color: ${palette('text', 3)};
      margin: 0 0 10px;
    }

    button {
      display: inline-block;
      margin-top: 15px;
      margin-bottom: 0;
      font-weight: 500;
      text-align: center;
      -ms-touch-action: manipulation;
      touch-action: manipulation;
      cursor: pointer;
      background-image: none;
      border: 0;
      white-space: nowrap;
      line-height: 1.5;
      padding: 0 20px;
      font-size: 13px;
      height: 42px;
      -webkit-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;
      position: relative;
      color: #ffffff;
      background-color: ${palette('pages', 2)};
      ${transition()};
      ${borderRadius('3px')};
      box-shadow: 0px 1px 5px 0px rgba(0, 0, 0, 0.2),
        0px 2px 2px 0px rgba(0, 0, 0, 0.14),
        0px 3px 1px -2px rgba(0, 0, 0, 0.12);

      a {
        width: 100%;
        height: 100%;
        color: #ffffff;
        text-decoration: none;
      }

      &:hover {
        background-color: ${palette('pages', 3)};

        a {
          text-decoration: none;
        }
      }

      &:focus {
        outline: 0;
        box-shadow: none;

        a {
          text-decoration: none;
        }
      }
    }
  }

  .mate500Artwork {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: ${props =>
      props['data-rtl'] === 'rtl' ? 'inherit' : '100px'};
    margin-right: ${props =>
      props['data-rtl'] === 'rtl' ? '100px' : 'inherit'};
    height: 400px;

    @media only screen and (max-width: 767px) {
      margin-left: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '0')};
      margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : 'inherit')};
    }

    img {
      max-width: 100%;
    }
  }
`;

export default WithDirection(FiveZeroFiveStyleWrapper);
