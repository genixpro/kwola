import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius } from '../../settings/style-util';
import WithDirection from '../../settings/withDirection';
import Icon from '../../components/uielements/icon';

const DeleteIcon = styled(Icon)`
  color: ${palette('grey', 4)};
  font-size: 18px;
  ${transition()};
`;

const WDSingleCardWrapper = styled.div`
  background-color: #ffffff;
  position: relative;
  box-shadow: ${palette('shadows', 1)};

  .cardImage {
    overflow: hidden;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    background-color: ${palette('grey', 1)};

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .cardContent {
    width: 100%;
    display: flex;
    flex-direction: column;
    margin: 0 15px;

    .cardTitle {
      font-size: 14px;
      font-weight: 500;
      color: ${palette('grey', 8)};
      margin: 0 0 5px;
    }

    .cardDate {
      font-size: 12px;
      font-weight: 400;
      color: ${palette('grey', 5)};
    }
  }

  .deleteBtn {
    width: 24px;
    height: 24px;
    background-color: transparent;
    flex-shrink: 0;
    padding: 0;
    border: 0;
    cursor: pointer;
    opacity: 0;
    ${transition()};

    &:hover {
      ${DeleteIcon} {
        color: ${palette('grey', 8)};
      }
    }
  }

  &:hover {
    .deleteBtn {
      opacity: 1;
    }
  }

  &.list {
    width: 100%;
    display: flex;
    padding: 15px;
    align-items: center;
    margin-bottom: 10px;

    .cardImage {
      width: 35px;
      height: 35px;
      display: -webkit-inline-flex;
      display: -ms-inline-flex;
      display: inline-flex;
      ${borderRadius('50%')};
    }
  }

  &.grid {
    .cardImage {
      width: 100%;
      height: 260px;
      display: flex;

      @media only screen and (min-width: 960px) {
        height: 330px;
      }
    }

    .cardContent {
      padding: 15px;
      margin: 0;
    }

    .deleteBtn {
      position: absolute;
      top: 0;
      right: ${props => (props['data-rtl'] === 'rtl' ? 'inherit' : '0')};
      left: ${props => (props['data-rtl'] === 'rtl' ? '0' : 'inherit')};
    }
  }
`;

const WDSortableCardWrapper = styled.div`
  padding: 50px 35px;
  * {
    box-sizing: border-box;
  }
  @media only screen and (max-width: 767px) {
    padding: 30px 20px;
  }

  .controlBar {
    width: 100%;
    display: flex;
    margin-bottom: 30px;
    align-items: center;

    button {
      margin-right: 10px;
      margin-bottom: 10px;
      span {
        font-size: 14px;
      }
      .material-icons {
        margin-right: 5px;
        color: #fff;
        font-size: 20px;
      }
    }

    @media only screen and (max-width: 667px) {
      align-items: flex-start;
      flex-direction: column;
    }

    > * {
      display: flex;
      align-items: center;

      .controlBtn {
        font-size: 12px;
        font-weight: 400;
        text-transform: uppercase;
        color: #ffffff;
        background-color: ${palette('primary', 0)};
        border: 0;
        outline: 0;
        display: -webkit-inline-flex;
        display: -ms-inline-flex;
        display: inline-flex;
        align-items: center;
        height: 36px;
        padding: 0 15px;
        margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : '10px')};
        margin-left: ${props => (props['data-rtl'] === 'rtl' ? '10px' : '0')};
        cursor: pointer;
        ${borderRadius('3px')};
        ${transition()};

        @media only screen and (max-width: 430px) {
          padding: 0 10px;
        }

        i {
          padding-right: ${props =>
            props['data-rtl'] === 'rtl' ? '0' : '10px'};
          padding-left: ${props =>
            props['data-rtl'] === 'rtl' ? '10px' : '0'};
        }

        &:last-child {
          margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : '0')};
          margin-left: ${props => (props['data-rtl'] === 'rtl' ? '0' : '0')};
        }

        &:hover {
          background-color: ${palette('primary', 1)};
        }
      }

      &.controlBtnGroup {
        flex-wrap: wrap;
        margin-left: ${props =>
          props['data-rtl'] === 'rtl' ? 'inherit' : 'auto'};
        margin-right: ${props =>
          props['data-rtl'] === 'rtl' ? 'auto' : 'inherit'};

        @media only screen and (max-width: 667px) {
          margin-left: ${props =>
            props['data-rtl'] === 'rtl' ? 'inherit' : '0'};
          margin-right: ${props =>
            props['data-rtl'] === 'rtl' ? '0' : 'inherit'};
          margin-top: 20px;
        }
        @media only screen and (max-width: 430px) {
          margin-top: 0px;
        }
        .ascendingBtn {
          background: #e0e0e0;
          span {
            color: rgba(0, 0, 0, 0.87);
          }
        }
        .shuffleBtn {
          background: #2196f3;
          span {
            color: #fff;
          }
        }
        .rotateBtn {
          background: #673ab7;
          span {
            color: #fff;
          }
        }
      }
    }
  }

  .addRemoveControlBar {
    width: 100%;
    display: flex;
    justify-content: center;
    margin-top: 30px;

    .controlBtnGroup {
      display: flex;
      align-items: center;
      button {
        margin-right: 10px;
        .material-icons {
          margin-right: 5px;
          color: #fff;
        }
      }
      .addBtn {
        .material-icons {
          color: ${palette('indigo', 6)};
        }
      }
      .removeBtn {
        .material-icons {
          color: #f50057;
        }
      }

      .controlBtn {
        font-size: 12px;
        font-weight: 400;
        padding: 0;
        text-transform: uppercase;
        color: #ffffff;
        background-color: ${palette('primary', 0)};
        border: 0;
        outline: 0;
        height: 36px;
        padding: 0 15px;
        margin-right: ${props => (props['data-rtl'] === 'rtl' ? '0' : '10px')};
        margin-left: ${props => (props['data-rtl'] === 'rtl' ? '10px' : '0')};
        cursor: pointer;
        ${borderRadius('3px')};
        ${transition()};
        .mateShuffleIcon {
          width: auto !important;
          height: auto !important;
          .material-icons {
            margin-right: 8px;
          }
        }

        i {
          padding-right: ${props =>
            props['data-rtl'] === 'rtl' ? '0' : '10px'};
          padding-left: ${props =>
            props['data-rtl'] === 'rtl' ? '10px' : '0'};
        }

        &:last-child {
          margin: 0;
        }

        &:hover {
          background-color: ${palette('primary', 1)};
        }
      }
    }
  }

  &.grid {
    .sortableCardsContainer {
      > div {
        width: 100%;
        display: flex;
        flex-flow: row wrap;
        padding: 0;

        ${WDSingleCardWrapper} {
          &.grid {
            width: calc(100% / 3 - 15px);
            display: flex;
            flex-direction: column;
            margin: 0 7px 15px;
            padding: 0;

            @media only screen and (max-width: 767px) {
              width: calc(100% / 2 - 10px);
              margin: 0 5px 10px;
            }

            @media only screen and (max-width: 480px) {
              width: 100%;
              margin-right: ${props =>
                props['data-rtl'] === 'rtl' ? 'inherit' : '0'};
              margin-left: ${props =>
                props['data-rtl'] === 'rtl' ? '0' : 'inherit'};
            }

            @media only screen and (min-width: 1400px) {
              width: calc(100% / 4 - 15px);
              margin: 0 7px 15px;
            }
          }
        }
      }
    }
  }
`;

const SingleCardWrapper = WithDirection(WDSingleCardWrapper);
const SortableCardWrapper = WithDirection(WDSortableCardWrapper);

export { SingleCardWrapper, SortableCardWrapper, DeleteIcon };
