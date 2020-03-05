import styled from 'styled-components';
import { palette } from 'styled-theme';
import Buttons from '../uielements/button';

const Button = styled(Buttons)``;

const SwitcherBtns = styled.div`
  width: 100%;
  padding: 20px 20px 15px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background-color: ${palette('grey', 2)};
  margin-bottom: 20px;

  h4 {
    font-size: 15px;
    font-weight: 500;
    color: ${palette('grey', 8)};
    line-height: 1;
    margin: 0 0 7px;
    padding: 0;
    text-transform: capitalize;
  }

  .themeSwitchBtnWrapper {
    width: 100%;
    display: flex;
    flex-flow: row wrap;
    align-items: center;

    ${Button} {
      width: 20px;
      height: 20px;
      display: flex;
      margin: ${props =>
        props['data-rtl'] === 'rtl' ? '5px 0 5px 10px' : '5px 10px 5px 0'};
      border: 1px solid #e4e4e4;
      outline: 0;
      padding: 0;
      min-width: auto;
      min-height: auto;
      background: none;
      justify-content: center;
      flex-shrink: 0;
      position: relative;
      cursor: pointer;
      border-radius: 3px;

      &.languageSwitch {
        border: 0;
        width: 30px;
        height: auto;

        &.selectedTheme {
          &:after {
            bottom: -7px;
            left: ${props =>
              props['data-rtl'] === 'rtl' ? 'inherit' : '11px'};
            right: ${props =>
              props['data-rtl'] === 'rtl' ? '11px' : 'inherit'};
          }
        }
      }

      img {
        width: 100%;
      }

      &.selectedTheme {
        &:after {
          content: '';
          display: -webkit-inline-flex;
          display: -ms-inline-flex;
          display: inline-flex;
          width: 0;
          height: 0;
          border-style: solid;
          border-width: 0 4px 4px 4px;
          border-color: transparent transparent ${palette('indigo', 4)}
            transparent;
          position: absolute;
          bottom: -7px;
          left: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '4px')};
          right: ${props => (props['data-rtl'] === 'rtl' ? '4px' : 'inherit')};
        }
      }
    }
  }
`;

export { Button, SwitcherBtns };
